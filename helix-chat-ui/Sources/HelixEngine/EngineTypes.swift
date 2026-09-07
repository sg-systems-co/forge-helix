import Foundation

public enum Role: String, Sendable, Codable {
    case user
    case assistant
}

public struct Message: Identifiable, Sendable, Equatable {
    public let id: UUID
    public let role: Role
    public var text: String
    /// Set once the turn finishes streaming.
    public var stats: TurnStats?

    public init(id: UUID = UUID(), role: Role, text: String, stats: TurnStats? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.stats = stats
    }
}

/// Per-turn timing, split so prefill (the path HELIX accelerates) is visible
/// separately from decode (which it deliberately does not touch).
public struct TurnStats: Sendable, Equatable {
    public var promptTokens: Int
    public var outputTokens: Int
    public var prefillSeconds: Double
    public var decodeSeconds: Double

    public init(promptTokens: Int, outputTokens: Int,
                prefillSeconds: Double, decodeSeconds: Double) {
        self.promptTokens   = promptTokens
        self.outputTokens   = outputTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds  = decodeSeconds
    }

    public var prefillTokensPerSecond: Double {
        prefillSeconds > 0 ? Double(promptTokens) / prefillSeconds : 0
    }
    public var decodeTokensPerSecond: Double {
        decodeSeconds > 0 ? Double(outputTokens) / decodeSeconds : 0
    }
}

/// What the engine emits while a turn runs.
public enum TurnEvent: Sendable {
    case prefilled(promptTokens: Int, seconds: Double)
    case token(String)
    case finished(TurnStats)
    case failed(String)
}

public enum EngineError: Error, LocalizedError {
    case modelLoadFailed(String)
    case contextInitFailed
    case tokenizationFailed
    case decodeFailed(Int32)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let p): return "Failed to load model at \(p)"
        case .contextInitFailed:      return "Failed to create llama context"
        case .tokenizationFailed:     return "Tokenization failed"
        case .decodeFailed(let c):    return "llama_decode failed (\(c))"
        }
    }
}

/// Which HELIX matmul path this model's SSM_SCAN shape actually resolves to.
///
/// Worth surfacing because it is not inferable from throughput: HELIX falls back
/// silently, so a model can be running the fp32 path while `GGML_HELIX_MPP=1` is
/// set and look identical from the outside.
public enum HelixBackend: String, Sendable {
    case mpp      = "MPP (M5 neural accelerators, bf16)"
    case sgmma    = "simdgroup_matrix (fp32)"
    case fallback = "ggml built-in (HELIX declined this shape)"
    case unavailable = "HELIX not present in this build"
}

/// Sampling chain parameters.
///
/// Ordering below follows llama.cpp's own default
/// (`common/common.h`: penalties -> top_k -> top_p -> temperature -> dist),
/// which matters: the repetition penalty has to see the raw logits, before
/// truncation narrows the candidate set.
public struct SamplingConfig: Sendable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int32
    /// 1.0 disables. Sub-3-bit models lock into phrase loops without this --
    /// it is the piece that was missing, not the sampling itself.
    public var penaltyRepeat: Float
    /// How far back the penalty looks.
    ///
    /// llama.cpp defaults this to 64, which is the wrong value here. Measured
    /// over a 4-turn conversation with 300-token replies, repetition *within*
    /// each reply is 0% in every configuration -- nothing loops internally. The
    /// pathology is the model restating its previous answers, and a 64-token
    /// window cannot reach past the current turn to see them:
    ///
    ///     window   whole-transcript repetition   worst repeated span
    ///     none                            50.2%            80 words
    ///     64                              45.1%            80
    ///     512                             51.9%            80
    ///     2048                            17.3%            57
    ///
    /// Raising the penalty instead of the window makes it worse (1.25 at 2048
    /// scored 27.9%), so this is the lever, not `penaltyRepeat`.
    public var penaltyLastN: Int32
    public var penaltyFrequency: Float
    public var penaltyPresence: Float
    public var seed: UInt32

    public init(temperature: Float = 0.7,
                topP: Float = 0.9,
                topK: Int32 = 40,
                penaltyRepeat: Float = 1.15,
                penaltyLastN: Int32 = 2048,
                penaltyFrequency: Float = 0.0,
                penaltyPresence: Float = 0.0,
                seed: UInt32 = 0xFFFFFFFF) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.penaltyRepeat = penaltyRepeat
        self.penaltyLastN = penaltyLastN
        self.penaltyFrequency = penaltyFrequency
        self.penaltyPresence = penaltyPresence
        self.seed = seed
    }
}

public struct EngineConfig: Sendable {
    /// $HELIX_MODEL_PATH, else `falcon-h1-7b-forge-v2.gguf` next to the binary.
    /// Download from https://huggingface.co/sgsystems/Falcon-H1-7B-FORGE-v2
    public static var defaultModelPath: String {
        if let env = ProcessInfo.processInfo.environment["HELIX_MODEL_PATH"], !env.isEmpty {
            return env
        }
        let exeDir = Bundle.main.executableURL?.deletingLastPathComponent()
            ?? URL(fileURLWithPath: ".")
        return exeDir.appendingPathComponent("falcon-h1-7b-forge-v2.gguf").path
    }

    public var modelPath: String
    public var contextLength: UInt32
    public var batchSize: UInt32
    public var ubatchSize: UInt32
    public var gpuLayers: Int32
    public var maxOutputTokens: Int
    public var sampling: SamplingConfig
    /// Prepended as a `system` turn on every render. Kept out of the message
    /// history so "New Chat" cannot drop it and multi-turn cannot duplicate it.
    public var systemPrompt: String?

    public init(
        // FORGE v2: ssm_out and ffn_down preserved at Q6_K, which is what
        // restores factual recall over the uniformly-ternary v1 build.
        //
        // Override with $HELIX_MODEL_PATH; otherwise looks for the GGUF beside
        // the executable, which is where a downloaded model naturally lands.
        modelPath: String = EngineConfig.defaultModelPath,
        // 4096, not 8192. Peak RSS measured at 3.59 / 3.68 / 3.86 GB for
        // n_ctx 2048 / 4096 / 8192; the iOS-style ~3.7 GB kill budget only
        // holds at 4096. Doubling the context costs ~180 MB of KV cache.
        contextLength: UInt32 = 4096,
        // Batch matches the context so a long prompt is prefilled in one
        // dispatch. ubatch of 512 keeps each SSM_SCAN call well above HELIX's
        // 64-token gate, which is where its chunk-parallel path pays.
        batchSize: UInt32 = 4096,
        ubatchSize: UInt32 = 512,
        gpuLayers: Int32 = 999,
        maxOutputTokens: Int = 512,
        sampling: SamplingConfig = SamplingConfig(),
        systemPrompt: String? = "You are a helpful, knowledgeable, and precise AI assistant."
    ) {
        self.modelPath = modelPath
        self.contextLength = contextLength
        self.batchSize = batchSize
        self.ubatchSize = ubatchSize
        self.gpuLayers = gpuLayers
        self.maxOutputTokens = maxOutputTokens
        self.sampling = sampling
        self.systemPrompt = systemPrompt
    }
}
