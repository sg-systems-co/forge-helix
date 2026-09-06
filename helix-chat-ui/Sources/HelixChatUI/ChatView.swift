import SwiftUI
import HelixEngine

struct ChatView: View {
    @State private var vm = ChatViewModel.shared
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            TopBar(
                title: vm.modelTitle,
                onNewChat: { vm.newChat() },
                backend: vm.helixBackend,
                templateActive: vm.templateActive
            )

            Group {
                if vm.messages.isEmpty {
                    EmptyState(title: vm.modelTitle, state: vm.loadState)
                } else {
                    MessageStream(messages: vm.messages)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Composer(
                text: $vm.input,
                canSend: vm.canSend,
                isGenerating: vm.isGenerating,
                onSend: { vm.send() },
                onStop: { vm.stop() }
            )
            .focused($inputFocused)
        }
        .background(Palette.background)
        // The reference is a light-mode design; letting it follow the system
        // would produce a look that was never specified.
        .preferredColorScheme(.light)
        .task { await vm.bootstrap() }
        .onAppear { inputFocused = true }
    }
}

// MARK: - Top bar

private struct TopBar: View {
    let title: String
    let onNewChat: () -> Void
    let backend: HelixBackend
    let templateActive: Bool

    var body: some View {
        ZStack {
            // Centered on the window, not on the space left over beside the
            // buttons, which is what keeps it centered as the title changes.
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Palette.title)
                .lineLimit(1)
                .truncationMode(.tail)
                .padding(.horizontal, 96)

            HStack(spacing: 18) {
                Spacer()
                Button(action: onNewChat) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 16, weight: .regular))
                }
                .buttonStyle(.plain)
                .help("New chat")

                Menu {
                    Section("HELIX backend") { Text(backend.rawValue) }
                    Section("Chat template") {
                        Text(templateActive ? "Model template applied" : "None — raw continuation")
                    }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 16, weight: .regular))
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Options")
            }
            .foregroundStyle(Palette.icon)
        }
        .padding(.horizontal, 20)
        .frame(height: 52)
    }
}

// MARK: - Empty state

private struct EmptyState: View {
    let title: String
    let state: ChatViewModel.LoadState

    var body: some View {
        VStack(spacing: 18) {
            Spacer()
            Image(systemName: "leaf")
                .font(.system(size: 26, weight: .light))
                .foregroundStyle(Palette.watermark)

            Text(title)
                .font(.system(size: 30, weight: .bold))
                .foregroundStyle(Palette.title)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 32)

            status
            Spacer()
            Spacer()   // biases the block above centre, as in the reference
        }
    }

    @ViewBuilder private var status: some View {
        switch state {
        case .idle, .loading:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Loading model…").font(.system(size: 12))
            }
            .foregroundStyle(Palette.subtle)
        case .ready:
            EmptyView()
        case .failed(let why):
            Text(why)
                .font(.system(size: 12))
                .foregroundStyle(.red.opacity(0.85))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }
}

// MARK: - Message stream

private struct MessageStream: View {
    let messages: [Message]
    private let anchor = "bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    ForEach(messages) { MessageRow(message: $0).id($0.id) }
                    Color.clear.frame(height: 1).id(anchor)
                }
                .padding(.horizontal, 22)
                .padding(.vertical, 22)
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
            }
            // Keyed on the streaming message's length so it follows tokens as
            // they land, not only when a turn ends.
            .onChange(of: messages.last?.text.count ?? 0) {
                withAnimation(.easeOut(duration: 0.12)) {
                    proxy.scrollTo(anchor, anchor: .bottom)
                }
            }
            .onChange(of: messages.count) { proxy.scrollTo(anchor, anchor: .bottom) }
        }
    }
}

private struct MessageRow: View {
    let message: Message

    var body: some View {
        switch message.role {
        case .user:
            HStack {
                Spacer(minLength: 56)
                Text(message.text)
                    .font(.system(size: 15))
                    .foregroundStyle(Palette.title)
                    .textSelection(.enabled)
                    .padding(.horizontal, 15)
                    .padding(.vertical, 10)
                    .background(Palette.field, in: .rect(cornerRadius: 20))
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 7) {
                // Left unbubbled and full width, which is what keeps a long
                // reply readable.
                Text(message.text.isEmpty ? " " : message.text)
                    .font(.system(size: 15))
                    .foregroundStyle(Palette.title)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if let s = message.stats {
                    // Prefill and decode kept apart: HELIX accelerates the first
                    // and deliberately leaves the second to ggml.
                    Text(String(format: "prefill %.0f tok/s (%d) · decode %.1f tok/s (%d)",
                                s.prefillTokensPerSecond, s.promptTokens,
                                s.decodeTokensPerSecond, s.outputTokens))
                        .font(.system(size: 10.5))
                        .foregroundStyle(Palette.subtle)
                }
            }
        }
    }
}

// MARK: - Composer

private struct Composer: View {
    @Binding var text: String
    let canSend: Bool
    let isGenerating: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            TextField("Make me a carrot cake", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .foregroundStyle(Palette.title)
                .lineLimit(1...6)
                .padding(.leading, 18)
                .padding(.trailing, 8)
                .onSubmit(onSend)

            // Sits inside the capsule, flush right.
            Button(action: isGenerating ? onStop : onSend) {
                Image(systemName: isGenerating ? "stop.fill" : "arrow.up")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 30, height: 30)
                    .background(Circle().fill(Palette.send))
                    // Solid black in every state, as in the reference. Dimmed
                    // rather than recoloured when there is nothing to send, so
                    // it still reads as inactive without changing the design.
                    .opacity((canSend || isGenerating) ? 1.0 : 0.28)
            }
            .buttonStyle(.plain)
            .disabled(!canSend && !isGenerating)
            .keyboardShortcut(.return, modifiers: [])
            .padding(.trailing, 6)
        }
        .padding(.vertical, 6)
        .background(Palette.field, in: .capsule)
        .frame(maxWidth: 640)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 22)
        .padding(.top, 8)
        .padding(.bottom, 18)
    }
}

// MARK: - Palette

/// Fixed light-mode values rather than semantic system colors: the reference is
/// a specific light design, and semantic colors would drift with appearance.
private enum Palette {
    static let background = Color.white
    static let title      = Color(red: 0.07, green: 0.07, blue: 0.08)
    static let icon       = Color(red: 0.28, green: 0.28, blue: 0.30)
    static let subtle     = Color(red: 0.60, green: 0.60, blue: 0.64)
    static let watermark  = Color(red: 0.55, green: 0.55, blue: 0.58)
    /// #F2F2F7 — iOS secondary system background.
    static let field      = Color(red: 0.949, green: 0.949, blue: 0.969)
    static let send       = Color.black
}
