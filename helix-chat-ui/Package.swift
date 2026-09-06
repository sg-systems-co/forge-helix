// swift-tools-version: 6.0
import PackageDescription
import Foundation

// Where the HELIX-accelerated llama.cpp was built. Override with
//   HELIX_LLAMA_BUILD=/path/to/llama.cpp/build swift build
let llamaBuild = ProcessInfo.processInfo.environment["HELIX_LLAMA_BUILD"]
    ?? "/Users/sebastiangrebe/Documents/Git/helix/third_party/llama.cpp/build"

let llamaInclude = "\(llamaBuild)/../include"
let ggmlInclude  = "\(llamaBuild)/../ggml/include"
let libDir       = "\(llamaBuild)/bin"

let package = Package(
    name: "HelixChatUI",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "HelixChatUI", targets: ["HelixChatUI"]),
        .executable(name: "helix-smoke", targets: ["helix-smoke"]),
        .library(name: "HelixEngine", targets: ["HelixEngine"]),
    ],
    targets: [
        // C surface: llama.cpp's own C API plus the HELIX introspection ABI.
        .target(
            name: "CLlamaBridge",
            cSettings: [
                .headerSearchPath("include"),
                .unsafeFlags(["-I\(llamaInclude)", "-I\(ggmlInclude)"]),
            ]
        ),

        // Swift actor wrapping load / prefill / decode.
        .target(
            name: "HelixEngine",
            dependencies: ["CLlamaBridge"],
            swiftSettings: [
                .unsafeFlags(["-Xcc", "-I\(llamaInclude)", "-Xcc", "-I\(ggmlInclude)"]),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L\(libDir)",
                    "-lllama", "-lggml", "-lggml-base", "-lggml-cpu",
                    "-lggml-metal", "-lggml-blas",
                    // The dylibs are not installed, so the binary needs to find
                    // them at their build location at run time. swiftc hands
                    // these to the driver, which needs -Xlinker for raw ld flags.
                    "-Xlinker", "-rpath", "-Xlinker", "\(libDir)",
                ]),
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate"),
            ]
        ),

        // Headless verification: loads, prefills, streams, prints the backend.
        .executableTarget(
            name: "helix-smoke",
            dependencies: ["HelixEngine"],
            linkerSettings: [
                .unsafeFlags(["-L\(libDir)", "-Xlinker", "-rpath", "-Xlinker", "\(libDir)"]),
            ]
        ),

        .executableTarget(
            name: "HelixChatUI",
            dependencies: ["HelixEngine"],
            swiftSettings: [
                .unsafeFlags(["-Xcc", "-I\(llamaInclude)", "-Xcc", "-I\(ggmlInclude)"]),
            ],
            linkerSettings: [
                .unsafeFlags(["-L\(libDir)", "-Xlinker", "-rpath", "-Xlinker", "\(libDir)"]),
                .linkedFramework("SwiftUI"),
                .linkedFramework("AppKit"),
            ]
        ),
    ]
)
