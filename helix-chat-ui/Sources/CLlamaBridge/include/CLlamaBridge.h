// CLlamaBridge.h -- the C surface Swift imports.
//
// llama.cpp is C++ internally but exposes a C API, so Swift can talk to it
// directly with no shim of our own. The only addition here is the HELIX
// introspection ABI, which is linked into libggml-metal but has no header in
// llama.cpp's include path -- redeclaring the handful of entry points we need
// is lighter than dragging HELIX's own headers into the app target.

#ifndef CLLAMA_BRIDGE_H
#define CLLAMA_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- HELIX introspection ---------------------------------------------------
//
// Mirrors include/helix/helix.h. Kept minimal on purpose: the app only needs to
// answer "which matmul backend will this model's SSM_SCAN actually take", which
// is the one question a user cannot answer by looking at throughput alone.

typedef struct helix_ctx helix_ctx;

typedef enum {
    HELIX_BACKEND_AUTO  = 0,
    HELIX_BACKEND_SGMMA = 1,  // simdgroup_matrix, fp32 (M1-M4, and M5 fallback)
    HELIX_BACKEND_MPP   = 2,  // MetalPerformancePrimitives, bf16 (M5 only)
    HELIX_BACKEND_REF   = 3,
} helix_backend_t;

typedef enum { HELIX_A_SCALAR_PER_HEAD = 0, HELIX_A_DIAG_PER_CHANNEL = 1 } helix_a_form_t;
typedef enum { HELIX_F32 = 0, HELIX_F16 = 1, HELIX_BF16 = 2 } helix_dtype_t;

// Mirrors helix_scan_desc in include/helix/helix.h. Declared in full rather
// than as opaque bytes so Swift can set `multipass` through the type system:
// the MPP trait requires it, and a probe that left it false would report SGMMA
// for a shape that actually runs on the neural accelerators.
typedef struct {
    int32_t n_tok, n_seq, n_head, head_dim, d_state, n_group;
    int64_t s_x[3], s_b[3], s_c[3], s_dt[2], s_s[3], s_y[3];
    int64_t s_a;
    int64_t seq_x, seq_b, seq_c, seq_dt, seq_s, seq_y;
    int32_t         chunk;
    helix_a_form_t  a_form;
    helix_dtype_t   io_dtype;
    helix_dtype_t   compute_dtype;
    helix_backend_t backend;
    bool            multipass;
    float           dt_min;
    float           dt_max;
} helix_scan_desc_t;

extern int  helix_ctx_create(helix_ctx **out, void *device, const char *metallib_path);
extern void helix_ctx_free(helix_ctx *ctx);
extern const char *helix_ctx_last_error(const helix_ctx *ctx);
extern int  helix_ctx_select_backend(helix_ctx *ctx, const helix_scan_desc_t *desc);
extern bool helix_scan_supported(helix_ctx *ctx, const helix_scan_desc_t *desc);
extern void helix_scan_desc_init(helix_scan_desc_t *desc,
                                 int32_t n_tok, int32_t n_seq, int32_t n_head,
                                 int32_t head_dim, int32_t d_state, int32_t n_group);

#ifdef __cplusplus
}
#endif

#endif  // CLLAMA_BRIDGE_H
