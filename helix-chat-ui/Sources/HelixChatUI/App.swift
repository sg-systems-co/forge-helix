import SwiftUI
import AppKit

@main
struct HelixChatApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("Helix Chat") {
            ChatView()
                .frame(minWidth: 520, minHeight: 420)
        }
        .defaultSize(width: 760, height: 720)
        .windowResizability(.contentSize)
        .commands { CommandGroup(replacing: .newItem) {} }
    }
}

/// A SwiftPM executable is not an app bundle, so AppKit starts it as a
/// background accessory: the window appears behind everything and never takes
/// focus. Promoting the activation policy is what makes `swift run` behave like
/// a normal app. An Xcode app target does not need this.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// ggml asserts that its Metal residency sets were all released when the
    /// device is destroyed. Quitting with the model still loaded aborts the
    /// process, so the engine is torn down before termination completes.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        Task { @MainActor in
            await ChatViewModel.shared.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
