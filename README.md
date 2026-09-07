# FORGE + HELIX

A 7B-parameter hybrid Mamba-2 assistant running natively on Apple Silicon in
**3.73 GB of RAM** at **75 tok/s** decode and **~1940 tok/s** prefill.

Two halves that solve different problems:

- **FORGE** — mixed-precision post-training quantization. Gets
  `Falcon-H1-7B-Instruct` from 15.5 GB to 3.48 GB (2.06 bpw average) without
  losing factual recall.
- **HELIX** — a Metal prefill kernel for the SSM scan, integrated into
  `llama.cpp` through a 42-line patch.

Plus **HelixChatUI**, a native SwiftUI client that ties them together.

---

## The physics

Edge inference is two separate constraints, and they pull in opposite
directions.

**Capacity.** A phone or laptop kills a process that exceeds roughly 3.7 GB
resident. A 7B model at fp16 is 15.5 GB. The gap is ~4.2x, and it has to come
out of the weights — which is a *capacity* problem: how much information can you
throw away before the model stops knowing things.

**Bandwidth.** Once the weights fit, throughput is bounded by how fast you move
them and how well the compute maps to the hardware. For a Mamba-2 hybrid, the
SSM scan is a sequential recurrence that mobile compilers do not map onto matrix
units at all — a *bandwidth and scheduling* problem, unrelated to how the weights
were compressed.

FORGE addresses the first. HELIX addresses the second. Neither helps with the
other, which is why they are separate.

### Where the memory actually goes

Measured on an M5 Max, `n_ubatch = 512`:

| `n_ctx` | peak RSS (llama-bench) |
|---|---|
| 2048 | 3.59 GB |
| **4096 (shipped)** | **3.68 GB** |
| 8192 | 3.86 GB |

The GGUF is 3.48 GB on disk; the rest is KV cache and compute buffers. **3.48 GB
is not the memory envelope** — doubling the context costs another ~180 MB, which
is the difference between running and being killed. The SwiftUI app measures
3.73 GB, about 50 MB above bare `llama-bench`.

Weights are `mmap`'d, so Metal maps the same pages under unified memory and there
is no host-to-device copy. That is what keeps a 3.48 GB model inside a 3.7 GB
process.

---

## FORGE: which tensors can be crushed

The headline result is not the bit width, it is **which tensors you are not
allowed to touch**.

Uniform ternary quantization produces a model that is fluent and confidently
wrong. In v1, with `ssm_out` crushed to ternary, the model listed **yeast and
rising dough as carrot cake ingredients** — carrot cake is chemically leavened,
so this is not a small error; the model had lost the association between a recipe
name and its method.

v2 keeps `ssm_out` and `ffn_down` at Q6_K and the failure goes away:

| | v1 (uniform ternary) | v2 (mixed) |
|---|---|---|
| size on disk | 3.25 GB | 3.48 GB |
| carrot cake, single turn | 3.0% repetition, span 7 | **0.0%, span 3** |
| 4 accumulated turns | 48.1% repetition, span 80 | **19.7%, span 43** |

The 230 MB those two tensor families cost is the entire difference between a
model that knows things and one that does not.

Configuration is recorded in the `.forge.json` sidecar shipped beside the GGUF
(`forge.solver.excluded_tensors = ffn_down,ssm_out`), so a build is reproducible
from the artifact alone.

See [`forge/`](forge/) and its [algorithm notes](forge/docs/algorithm.md).

---

## HELIX: chunk-parallel SSM scan

`llama.cpp` already ships a chunked SSD Metal kernel. HELIX rewrites the
decomposition:

| pass | |
|---|---|
| A `helix_chunk_state` | every chunk computes its own `dS` independently |
| B `helix_state_scan` | associative scan over `(decay, dS)` → per-chunk initial states |
| C `helix_chunk_out` | every chunk computes its output from its initial state |

A and C are parallel across chunks; upstream walks chunks serially inside one
threadgroup per head.

**Isolated op speedup on this model's shape** (`d_state 256, head_dim 128`):
**2.90x** at L=2048, **3.06x** at L=8192 — upstream falls back to its scalar
kernel here, because its fast path requires `d_inner == 64`.

**End-to-end on Falcon-H1-7B-FORGE-v2:**

| | HELIX off | HELIX on | |
|---|---|---|---|
| pp2048 | 1428.6 t/s | **1941.8 t/s** | 1.36x |
| pp8192 | 1260.5 t/s | **1623.9 t/s** | 1.29x |
| tg64 | 74.4 t/s | 74.6 t/s | 1.00x |

The gap between 3x on the op and 1.3x end-to-end is Amdahl: `SSM_SCAN` is roughly
a quarter of prefill time in a 7B hybrid; the rest is attention, FFN and
dequantization.

**HELIX is a prefill accelerator.** Decode is untouched by design — a
single-token step is bandwidth-bound, not a chunked-scan problem, and below 64
tokens HELIX declines and ggml's kernel runs. Worst case across the shape matrix
is 1.00x.

### On precision, measured rather than assumed

The plan assumed bf16 matrix ops would be ~2x fp32. On M5 that is false:

| path | operands | TFLOP/s |
|---|---|---|
| `simdgroup_matrix` | f32 | 16.12 |
| `simdgroup_matrix` | bf16 | 10.88 |
| plain vector `fma` | f32 | 14.18 |
| MPP `matmul2d` | f32 | 15.69 |
| MPP `matmul2d` | **bf16** | **65.68** |

`simdgroup_matrix` at fp32 barely clears plain vector FMA — it is not the matrix
hardware. The M5 neural accelerators are reachable only through
`MetalPerformancePrimitives`, and only with **both** operands bf16
(`bf16 x f32` measures 15.65, indistinguishable from f32).

That path is opt-in (`GGML_HELIX_MPP=1`) because bf16 cannot meet ggml's 2e-7 op
tolerance. **It does not engage on this model** — Falcon-H1's `d_state` is 256
while the MPP kernel is compiled for 128, so it falls back to fp32. The app
reports which path is live under the `...` menu rather than letting you assume.

See [`helix/`](helix/), [`docs/PRECISION.md`](helix/docs/PRECISION.md), and the
[upstream patch](helix/integration/ggml/llama.cpp-helix.patch).

---

## HelixChatUI

Native SwiftUI client. `swift run -c release HelixChatUI`.

### Teardown: two separate Swift 6 problems

Worth documenting because they look like one bug and are not.

**1. The actor could not free its own C pointers.** An actor's `deinit` is
nonisolated and may not touch non-Sendable stored properties. `isolated deinit`
can — but it raises the deployment floor to macOS 15.4 *and* trips
`error: circular reference` in the Swift 6.3.3 compiler under release
whole-module optimization (debug builds and `-no-whole-module-optimization` are
fine). The handles therefore live in a small boxed class whose own plain `deinit`
frees them:

```swift
private final class Handles: @unchecked Sendable {
    var model: OpaquePointer?
    var ctx: OpaquePointer?
    var sampler: UnsafeMutablePointer<llama_sampler>?
    deinit { /* llama_sampler_free / llama_free / llama_model_free */ }
}
```

**2. That was not enough, and the process still aborted on quit.**

```
ggml-metal-device.m:1021: GGML_ASSERT([rsets->data count] == 0) failed
```

ggml registers a Metal residency set per device and asserts at
static-destructor time that all of them were released. The engine is held in a
global or a long-lived view model, and **Swift does not deinitialise globals at
exit** — so the model was still alive when ggml tore the device down. The fix is
an explicit idempotent `LlamaEngine.shutdown()`, called from
`applicationShouldTerminate` and from the headless harness.

Boxing fixed the compiler problem. Only explicit shutdown fixed the crash.

### Sampling

```
penalties(repeat 1.15, window 2048) → top_k 40 → top_p 0.9 → temp 0.7 → dist
```

Order matches llama.cpp's own default so the penalty sees raw logits before
top-k/top-p truncate the candidate set.

**`penalty_last_n` is the parameter that matters, and llama.cpp's default of 64
is wrong here.** Measured over a 4-turn conversation with 300-token replies,
repetition *within* each reply is 0% in every configuration — nothing loops
internally. The pathology is the model restating previous answers, and a
64-token window cannot reach back far enough to see them:

| `penalty_last_n` | whole-transcript repetition | worst span |
|---|---|---|
| no penalty | 50.2% | 80 words |
| 64 (llama.cpp default) | 45.1% | 80 |
| 512 | 51.9% | 80 |
| **2048 (shipped)** | **17.3%** | **57** |

Raising the *strength* instead makes it worse (`penalty_repeat` 1.25 at window
2048 scored 27.9%). Shipping the parameter triple without the window reproduces
the bug.

See [`helix-chat-ui/README.md`](helix-chat-ui/README.md).

---

## Build

```sh
# 1. HELIX kernels
cd helix && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build          # 216 parity + 8 drift cases

# 2. llama.cpp with the HELIX patch
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
cd third_party/llama.cpp && git checkout $(cat ../../integration/ggml/PINNED_SHA)
git apply ../../integration/ggml/llama.cpp-helix.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_METAL=ON -DGGML_METAL_HELIX=ON \
      -DHELIX_ROOT=$PWD/../.. -DHELIX_BUILD=$PWD/../../build
cmake --build build

# 3. The app
cd ../../../helix-chat-ui && swift run -c release HelixChatUI
```

Requires macOS 14+, Xcode 26 with the Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`).

## Model

[`sgsystems/Falcon-H1-7B-FORGE-v2`](https://huggingface.co/sgsystems/Falcon-H1-7B-FORGE-v2)
— see [`deploy/MODEL_CARD.md`](deploy/MODEL_CARD.md).

## Honest limitations

- **Cross-turn repetition is reduced, not eliminated.** v2 with a 2048-token
  penalty window still measures 19.7% over four accumulated turns, with 43-word
  verbatim spans.
- **Decode is unimproved and will stay that way** without a separate fused
  single-step kernel.
- **Apple-only.** The 3-pass decomposition is portable; this implementation is
  not.
- **The MPP/neural-accelerator path does not run on this model** (`d_state` 256
  vs a kernel compiled for 128).
- **macOS only in practice.** The engine is portable Swift over llama.cpp's C
  API, but the UI is AppKit and a 3.48 GB resident model exceeds what most
  iPhones permit.
