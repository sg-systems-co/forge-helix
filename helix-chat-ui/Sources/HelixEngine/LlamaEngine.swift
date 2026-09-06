import Foundation
import CLlamaBridge

/// Owns the llama.cpp model and context.
///
/// An actor because llama_context is not thread-safe and a chat UI will happily
/// fire overlapping requests: serialising at the type level is cheaper than
/// auditing every call site. Generation runs on a detached task so the actor is
/// never blocked for the length of a turn.
/// Owns the llama.cpp C handles so they can be released from a plain deinit.
///
/// An actor's deinit is nonisolated and may not touch non-Sendable stored
/// properties. `isolated deinit` can, but it raises the deployment floor to
/// macOS 15.4 *and* trips a "circular reference" compiler error under release
/// whole-module optimization in Swift 6.3.3. Boxing the handles sidesteps both:
/// the box is released when the actor is, and its own deinit is free to touch
/// its own storage.
private final class Handles: @unchecked Sendable {
    var model: OpaquePointer?
    var ctx: OpaquePointer?
    var vocab: OpaquePointer?
    var sampler: UnsafeMutablePointer<llama_sampler>?

    deinit {
        if let sampler { llama_sampler_free(sampler) }
        if let ctx     { llama_free(ctx) }
        if let model   { llama_model_free(model) }
    }
}

public actor LlamaEngine {
    private let h = Handles()

    private let config: EngineConfig
    private var loaded = false

    /// Exactly the tokens currently resident in the KV cache.
    ///
    /// Kept so a follow-up turn only prefills what the chat template actually
    /// added, rather than re-running the whole transcript. Comparing token
    /// arrays rather than tracking a counter also survives templates that
    /// rewrite earlier turns instead of appending to them.
    private var cachedTokens: [llama_token] = []

    /// Conversation so far, in template terms.
    private var history: [(role: String, content: String)] = []

    /// The model's own chat template from GGUF metadata, if it ships one.
    private var chatTemplate: UnsafePointer<CChar>?

    /// Set when the consumer stops reading the stream.
    ///
    /// The generation task is detached, so cancelling the *consumer's* task does
    /// not cancel it -- without this the Stop button only stopped the UI from
    /// listening while the engine kept decoding to the token limit, burning GPU
    /// and making the next turn queue behind a reply nobody wanted.
    private var cancelled = false

    public init(config: EngineConfig = EngineConfig()) {
        self.config = config
    }

    // MARK: - Loading

    public func load() throws {
        guard !loaded else { return }

        guard FileManager.default.fileExists(atPath: config.modelPath) else {
            throw EngineError.modelLoadFailed(config.modelPath)
        }

        llama_backend_init()

        var mparams = llama_model_default_params()
        // Everything on the GPU. Under unified memory the weights are mmap'd and
        // Metal maps the same pages, so there is no host->device copy -- which
        // is what keeps a 3 GB model inside a 3.6 GB resident footprint.
        mparams.n_gpu_layers = config.gpuLayers
        mparams.load_mode    = LLAMA_LOAD_MODE_MMAP

        guard let m = llama_model_load_from_file(config.modelPath, mparams) else {
            throw EngineError.modelLoadFailed(config.modelPath)
        }
        h.model = m
        h.vocab = llama_model_get_vocab(m)

        var cparams = llama_context_default_params()
        cparams.n_ctx    = config.contextLength
        cparams.n_batch  = config.batchSize
        cparams.n_ubatch = config.ubatchSize

        guard let c = llama_init_from_model(m, cparams) else {
            llama_model_free(m)
            h.model = nil
            throw EngineError.contextInitFailed
        }
        h.ctx = c

        // Chain order mirrors llama.cpp's own default: penalties first, so the
        // repetition penalty sees the full distribution before top-k/top-p
        // truncate it. Applying it after truncation would let a looping token
        // survive simply by being the only candidate left.
        let sp = config.sampling
        var sparams = llama_sampler_chain_default_params()
        sparams.no_perf = true
        let chain = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            llama_vocab_n_tokens(h.vocab),
            sp.penaltyLastN, sp.penaltyRepeat, sp.penaltyFrequency, sp.penaltyPresence))
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.topK))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.topP, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temperature))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(sp.seed))
        h.sampler = chain

        // Falcon-H1 ships a ChatML template in its GGUF metadata. Without
        // applying it the model just continues text instead of answering, which
        // is what the first version of this engine did.
        chatTemplate = llama_model_chat_template(m, nil)

        loaded = true
    }

    public var isLoaded: Bool { loaded }

    /// True when the model carries its own chat template.
    public var hasChatTemplate: Bool { chatTemplate != nil }

    /// Releases the model, context and sampler.
    ///
    /// Must be called before the process exits. ggml registers a Metal
    /// residency set per device and asserts at static-destructor time that all
    /// of them were released:
    ///
    ///     ggml-metal-device.m:1021: GGML_ASSERT([rsets->data count] == 0)
    ///
    /// Relying on `Handles.deinit` is not enough, because the engine is
    /// normally held in a global or a long-lived view model and Swift does not
    /// deinitialise globals at exit -- so the model is still alive when ggml
    /// tears the device down, and the process aborts. Idempotent.
    public func shutdown() {
        if let s = h.sampler { llama_sampler_free(s); h.sampler = nil }
        if let c = h.ctx     { llama_free(c);         h.ctx = nil }
        if let m = h.model   { llama_model_free(m);   h.model = nil }
        h.vocab = nil
        chatTemplate = nil
        cachedTokens.removeAll()
        history.removeAll()
        loaded = false
    }

    /// Drop the KV cache and start a fresh conversation.
    public func reset() {
        guard let ctx = h.ctx else { return }
        llama_memory_clear(llama_get_memory(ctx), true)
        cachedTokens.removeAll()
        history.removeAll()
    }

    // MARK: - Chat templating

    /// Renders the conversation through the model's chat template.
    ///
    /// `add_ass` opens the assistant turn, so generation starts inside the
    /// reply rather than by predicting the next role header.
    private func renderPrompt() -> String? {
        guard let tmpl = chatTemplate else { return nil }

        // llama_chat_message holds borrowed C strings, so the backing buffers
        // have to outlive the call -- hence the explicit strdup/free rather
        // than passing Swift string temporaries.
        var cstrings: [UnsafeMutablePointer<CChar>] = []
        defer { cstrings.forEach { free($0) } }

        var msgs: [llama_chat_message] = []

        // System turn first, and rendered fresh each time rather than stored in
        // history -- so it survives "New Chat" and cannot be duplicated across
        // turns.
        if let sys = config.systemPrompt, !sys.isEmpty {
            let r = strdup("system")!
            let c = strdup(sys)!
            cstrings.append(r); cstrings.append(c)
            msgs.append(llama_chat_message(role: UnsafePointer(r), content: UnsafePointer(c)))
        }

        for turn in history {
            let r = strdup(turn.role)!
            let c = strdup(turn.content)!
            cstrings.append(r); cstrings.append(c)
            msgs.append(llama_chat_message(role: UnsafePointer(r), content: UnsafePointer(c)))
        }

        // Recommended sizing is 2x the total content length; grow once if the
        // template expands more than that.
        let contentBytes = history.reduce(config.systemPrompt?.utf8.count ?? 0) {
            $0 + $1.role.utf8.count + $1.content.utf8.count
        }
        var cap = max(1024, contentBytes * 4)
        for _ in 0..<2 {
            var buf = [CChar](repeating: 0, count: cap)
            let n = llama_chat_apply_template(tmpl, msgs, msgs.count, true, &buf, Int32(cap))
            if n < 0 { return nil }
            if Int(n) <= cap {
                return String(decoding: buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            cap = Int(n) + 64
        }
        return nil
    }

    // MARK: - Generation

    /// Streams one assistant turn.
    ///
    /// Prefill and decode are timed separately: prefill is the phase HELIX
    /// accelerates, decode is explicitly left to ggml's kernel, and folding them
    /// into one number would hide both facts.
    public func generate(prompt: String) -> AsyncStream<TurnEvent> {
        AsyncStream { continuation in
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason {
                    Task { await self?.requestCancel() }
                }
            }
            Task.detached { [weak self] in
                guard let self else { continuation.finish(); return }
                await self.clearCancel()
                await self.runTurn(prompt: prompt, into: continuation)
                continuation.finish()
            }
        }
    }

    private func requestCancel() { cancelled = true }
    private func clearCancel()   { cancelled = false }

    private func runTurn(prompt: String, into cont: AsyncStream<TurnEvent>.Continuation) {
        guard loaded, let ctx = h.ctx, let vocab = h.vocab, let sampler = h.sampler else {
            cont.yield(.failed("engine not loaded"))
            return
        }

        // --- apply the chat template ---------------------------------------
        history.append((role: "user", content: prompt))

        let rendered = renderPrompt()
        // A model with no template falls back to raw continuation, which is
        // wrong for an assistant but better than refusing to run.
        let promptText = rendered ?? prompt
        if ProcessInfo.processInfo.environment["HELIX_DEBUG_PROMPT"] == "1" {
            FileHandle.standardError.write(Data("\n=== rendered prompt ===\n\(promptText)\n=== end ===\n".utf8))
        }

        var tokens: [llama_token]
        do {
            // The template emits its own BOS, so add_special would double it.
            tokens = try tokenize(promptText, addSpecial: rendered == nil && cachedTokens.isEmpty)
        } catch {
            cont.yield(.failed(error.localizedDescription))
            history.removeLast()
            return
        }
        guard !tokens.isEmpty else {
            cont.yield(.failed("empty prompt"))
            history.removeLast()
            return
        }
        guard tokens.count < Int(config.contextLength) else {
            cont.yield(.failed("prompt exceeds context window"))
            history.removeLast()
            return
        }

        // --- reuse whatever prefix is already in the KV cache ---------------
        var reuse = 0
        while reuse < cachedTokens.count, reuse < tokens.count,
              cachedTokens[reuse] == tokens[reuse] {
            reuse += 1
        }
        if reuse < cachedTokens.count {
            // The template rewrote history; drop the diverging tail.
            llama_memory_seq_rm(llama_get_memory(ctx), 0, Int32(reuse), -1)
        }
        cachedTokens = tokens

        var pending = Array(tokens[reuse...])
        guard !pending.isEmpty else {
            cont.yield(.failed("nothing new to prefill"))
            return
        }

        // --- prefill -------------------------------------------------------
        //
        // The whole prompt goes in as one batch. llama.cpp splits it into
        // n_ubatch pieces internally, and each piece becomes one SSM_SCAN call
        // -- which is what puts HELIX's chunk-parallel path on the critical
        // path rather than its short-sequence fallback.
        let t0 = DispatchTime.now()
        let promptTokens = pending.count

        var rc: Int32 = 0
        pending.withUnsafeMutableBufferPointer { buf in
            var batch = llama_batch_get_one(buf.baseAddress, Int32(buf.count))
            rc = llama_decode(ctx, batch)
            _ = batch
        }
        guard rc == 0 else {
            cont.yield(.failed("prefill failed (llama_decode \(rc))"))
            return
        }

        let tPrefill = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
        cont.yield(.prefilled(promptTokens: promptTokens, seconds: tPrefill))

        // --- decode --------------------------------------------------------
        let t1 = DispatchTime.now()
        var produced = 0
        var reply = ""
        var pieceBuf = [CChar](repeating: 0, count: 256)

        while produced < config.maxOutputTokens {
            if cancelled { break }
            let id = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, id) { break }

            let n = llama_token_to_piece(vocab, id, &pieceBuf, Int32(pieceBuf.count), 0, false)
            if n > 0 {
                let piece = pieceBuf.withUnsafeBufferPointer { p -> String in
                    let bytes = UnsafeRawBufferPointer(start: p.baseAddress, count: Int(n))
                    return String(decoding: bytes.bindMemory(to: UInt8.self), as: UTF8.self)
                }
                reply += piece
                cont.yield(.token(piece))
            }

            llama_sampler_accept(sampler, id)
            produced += 1

            var next = id
            let step = withUnsafeMutablePointer(to: &next) { ptr -> Int32 in
                var batch = llama_batch_get_one(ptr, 1)
                let r = llama_decode(ctx, batch)
                _ = batch
                return r
            }
            guard step == 0 else {
                cont.yield(.failed("decode failed (llama_decode \(step))"))
                return
            }
            cachedTokens.append(id)
        }

        // Record the reply so the next turn's template renders real history.
        history.append((role: "assistant", content: reply))

        let tDecode = Double(DispatchTime.now().uptimeNanoseconds - t1.uptimeNanoseconds) / 1e9
        cont.yield(.finished(TurnStats(
            promptTokens: promptTokens,
            outputTokens: produced,
            prefillSeconds: tPrefill,
            decodeSeconds: tDecode
        )))
    }

    private func tokenize(_ text: String, addSpecial: Bool) throws -> [llama_token] {
        guard let vocab = h.vocab else { throw EngineError.tokenizationFailed }
        let utf8 = Array(text.utf8)
        // Upper bound: one token per byte plus room for specials.
        var out = [llama_token](repeating: 0, count: utf8.count + 8)
        let n = utf8.withUnsafeBufferPointer { bytes -> Int32 in
            bytes.baseAddress!.withMemoryRebound(to: CChar.self, capacity: bytes.count) { cstr in
                llama_tokenize(vocab, cstr, Int32(bytes.count),
                               &out, Int32(out.count), addSpecial, true)
            }
        }
        guard n >= 0 else { throw EngineError.tokenizationFailed }
        return Array(out.prefix(Int(n)))
    }
}
