import Foundation
import Observation
import HelixEngine

@MainActor
@Observable
final class ChatViewModel {
    /// Shared so the app delegate can shut the engine down on quit; the app has
    /// a single chat window and a single 3.5 GB model, so one instance is the
    /// honest model of the situation rather than a convenience.
    static let shared = ChatViewModel()

    var messages: [Message] = []
    var input: String = ""
    var isGenerating = false
    var loadState: LoadState = .idle
    var helixBackend: HelixBackend = .unavailable
    var templateActive = false

    enum LoadState: Equatable {
        case idle, loading, ready, failed(String)
    }

    private let engine: LlamaEngine
    private let config: EngineConfig
    private var turn: Task<Void, Never>?

    init(config: EngineConfig = EngineConfig()) {
        self.config = config
        self.engine = LlamaEngine(config: config)
    }

    var modelTitle: String { "Falcon-H1-7B (FORGE + HELIX)" }

    var canSend: Bool {
        loadState == .ready && !isGenerating &&
        !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Seeds a conversation so the populated state can be inspected without a
    /// model load. UI-development aid; never on in a normal run.
    func seedPreview() {
        messages = [
            Message(role: .user, text: "Make me a carrot cake. List the ingredients only, no steps."),
            Message(role: .assistant,
                    text: "Ingredients:\n\n- 2 cups flour\n- 2 cups grated carrots\n- 1 cup sugar\n- 3 eggs\n- 1 tsp cinnamon\n- 1 tsp baking powder",
                    stats: TurnStats(promptTokens: 23, outputTokens: 120,
                                     prefillSeconds: 0.010, decodeSeconds: 1.60)),
        ]
        input = "How many eggs?"
    }

    func bootstrap() async {
        if ProcessInfo.processInfo.environment["HELIX_UI_PREVIEW"] == "1" {
            seedPreview()
            helixBackend = HelixProbe.backend(for: .falconH1_7B)
            templateActive = true
            loadState = .ready
            return
        }
        guard loadState == .idle else { return }
        loadState = .loading

        // Ask HELIX which path this model's SSM shape resolves to before any
        // inference runs, so the answer is not confounded by throughput.
        helixBackend = HelixProbe.backend(for: .falconH1_7B)

        do {
            try await engine.load()
            templateActive = await engine.hasChatTemplate
            loadState = .ready
        } catch {
            loadState = .failed(error.localizedDescription)
        }
    }

    func newChat() {
        turn?.cancel()
        turn = nil
        isGenerating = false
        Task { await engine.reset() }
        messages.removeAll()
    }

    func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating else { return }
        input = ""

        messages.append(Message(role: .user, text: text))
        let replyIndex = messages.count
        messages.append(Message(role: .assistant, text: ""))
        isGenerating = true

        turn = Task { [engine] in
            for await event in await engine.generate(prompt: text) {
                if Task.isCancelled { break }
                guard replyIndex < messages.count else { break }
                switch event {
                case .prefilled:
                    break  // folded into .finished
                case .token(let piece):
                    // Appending on the main actor per token is what makes the
                    // stream visibly incremental. At ~75 tok/s that is well
                    // inside what SwiftUI can coalesce.
                    messages[replyIndex].text += piece
                case .finished(let stats):
                    messages[replyIndex].stats = stats
                case .failed(let why):
                    messages[replyIndex].text += "\n\n[error: \(why)]"
                }
            }
            isGenerating = false
        }
    }

    /// Called on app termination. Without this, ggml aborts on quit when its
    /// Metal device is destroyed with the model still alive.
    func shutdown() async {
        turn?.cancel()
        turn = nil
        await engine.shutdown()
    }

    func stop() {
        turn?.cancel()
        turn = nil
        isGenerating = false
    }
}
