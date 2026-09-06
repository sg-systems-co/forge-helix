import Foundation
import CLlamaBridge

/// Reports which HELIX matmul path a given SSM shape actually resolves to.
///
/// This exists because HELIX fails *soft*: when a shape falls outside what a
/// backend supports it silently falls back, so `GGML_HELIX_MPP=1` can be set,
/// the app can be visibly fast, and the M5 neural accelerators can still be
/// completely idle. Falcon-H1-7B is exactly that case -- its `d_state` is 256
/// while the MPP kernel is compiled for 128, so it runs the fp32 path.
///
/// Throughput alone cannot distinguish those, which is why the app asks
/// directly rather than inferring.
public enum HelixProbe {
    /// SSM geometry, read from GGUF metadata.
    public struct Shape: Sendable {
        public var dState: Int32
        public var headDim: Int32
        public var nHead: Int32
        public var nGroup: Int32
        public var nTokens: Int32

        public init(dState: Int32, headDim: Int32, nHead: Int32,
                    nGroup: Int32, nTokens: Int32 = 512) {
            self.dState = dState
            self.headDim = headDim
            self.nHead = nHead
            self.nGroup = nGroup
            self.nTokens = nTokens
        }

        /// Falcon-H1-7B-Instruct: ssm.state_size 256, inner 3072, 24 heads.
        public static let falconH1_7B = Shape(dState: 256, headDim: 128, nHead: 24, nGroup: 1)
    }

    public static func backend(for shape: Shape) -> HelixBackend {
        var ctx: OpaquePointer?
        guard helix_ctx_create(&ctx, nil, nil) == 0, let ctx else {
            return .unavailable
        }
        defer { helix_ctx_free(ctx) }

        var desc = helix_scan_desc_t()
        helix_scan_desc_init(&desc, shape.nTokens, 1, shape.nHead,
                             shape.headDim, shape.dState, shape.nGroup)
        // The ggml bridge always requests the 3-pass path, and the MPP trait
        // requires it. Asking with multipass=false would report SGMMA for a
        // shape that really runs on the neural accelerators.
        desc.multipass = true

        guard helix_scan_supported(ctx, &desc) else { return .fallback }

        switch helix_ctx_select_backend(ctx, &desc) {
        case Int32(HELIX_BACKEND_MPP.rawValue):   return .mpp
        case Int32(HELIX_BACKEND_SGMMA.rawValue): return .sgmma
        default:                                  return .fallback
        }
    }
}
