# helix-chat-ui

A native SwiftUI chat client for a locally quantized **Falcon-H1-7B-Instruct**
(2.06 bpw, 3.02 GiB) running on the HELIX-accelerated `llama.cpp` engine.

Holds **3.68 GB resident** and streams at **~75 tok/s** decode on an M5 Max.
First load is ~12s from cold page cache and ~0.2s once the file is warm --
`mmap` means the second launch pays almost nothing.

## Layout

```
Sources/
  CLlamaBridge/    llama.cpp's C API + the HELIX introspection ABI
  HelixEngine/     LlamaEngine actor, HelixProbe, streaming types
  HelixChatUI/     SwiftUI app
  helix-smoke/     headless load/prefill/decode check
```

## Build and run

Requires a built HELIX-enabled llama.cpp (see `../helix/integration/ggml/README.md`).

```sh
swift build -c release
swift run -c release HelixChatUI
```

Point the package at a different llama.cpp build with
`HELIX_LLAMA_BUILD=/path/to/llama.cpp/build swift build`.

Headless check, no UI:

```sh
swift run helix-smoke
```

```
HELIX backend for Falcon-H1-7B: simdgroup_matrix (fp32)
model loaded in 0.20s
chat template present: true

Q: Make me a carrot cake. List the ingredients only, no steps.
A: [prefill] 23 new tok in 0.005s = 4319 tok/s
Ingredients:
- 2 cups flour ...
[decode] 120 tok at 75.2 tok/s

Q: What is the capital of France? Answer in one word.
A: The capital of France is Paris.
[decode] 7 tok at 38.2 tok/s
```

The model path defaults to
`/Users/sebastiangrebe/Documents/Git/forge/out/falcon-h1-7b-forge-v2.gguf`;
override via `EngineConfig(modelPath:)`.

## Verifying HELIX is actually active

**Do not infer this from throughput.** HELIX fails *soft*: when a shape falls
outside what a backend supports it silently falls back, so `GGML_HELIX_MPP=1`
can be set, the app can be fast, and the M5 neural accelerators can still be
completely idle.

The app asks HELIX directly. `HelixProbe.backend(for:)` builds the model's SSM
descriptor and calls `helix_ctx_select_backend`, and the answer is shown under
the `...` menu. Three possible readings:

| shown | meaning |
|---|---|
| `MPP (M5 neural accelerators, bf16)` | the fast path is live |
| `simdgroup_matrix (fp32)` | HELIX is running, but on the portable path |
| `ggml built-in (HELIX declined this shape)` | HELIX is not handling this op |

**Falcon-H1-7B reports `simdgroup_matrix (fp32)`, and that is correct.** Its
`ssm.state_size` is 256 while HELIX's MPP kernel is compiled for `k = 128`, so
MPP declines and the fp32 path runs. Setting `GGML_HELIX_MPP=1` for this model
has no effect. Models with `d_state = 128` (e.g. Falcon-H1-0.5B) do take the MPP
path.

To confirm HELIX is contributing at all, compare prefill against the stock
kernels:

```sh
GGML_HELIX_DISABLE=1 swift run helix-smoke   # ggml only
swift run helix-smoke                        # HELIX
```

On this model the SSM_SCAN op itself is ~2.9-3.1x faster, which is ~1.2x
end-to-end -- SSM_SCAN is only about a quarter of prefill time in a 7B hybrid.

## Chat templating

Falcon-H1 ships a ChatML template in its GGUF metadata.
`llama_model_chat_template` reads it and `llama_chat_apply_template` renders the
conversation through it, with `add_ass: true` so generation begins inside the
assistant turn rather than by predicting the next role header.

This is what makes the model behave as an assistant. Without it the model
continues the prompt as text; with it, "List the ingredients only, no steps"
produces an ingredient list and no steps.

`HELIX_DEBUG_PROMPT=1` dumps the rendered prompt to stderr, which is the fastest
way to tell a templating bug from a model-quality one:

```
<|im_start|>user
Make me a carrot cake. List the ingredients only, no steps.<|im_end|>
<|im_start|>assistant
Ingredients: ...<|im_end|>
<|im_start|>user
Repeat my first message back to me exactly.<|im_end|>
<|im_start|>assistant
```

### KV reuse across turns

Each turn compares the newly rendered token sequence against what is already in
the KV cache, truncates at the first divergence (`llama_memory_seq_rm`) and
prefills only the remainder. Comparing token arrays rather than tracking a
counter also survives templates that rewrite earlier turns instead of appending.

**Reuse is opportunistic, not guaranteed.** History is stored as *text* and
re-rendered through the template each turn, and detokenize -> retokenize is not
always the identity — especially when a reply was cut at the token limit
mid-word. Observed second-turn prefill has ranged from 18 tokens (near-perfect
reuse) to 115 (divergence early in the assistant turn). It is always correct,
just sometimes slower than it needs to be. Storing the generated token ids
alongside the text would make reuse exact.

## Engine notes

- **`LlamaEngine` is an actor.** `llama_context` is not thread-safe and a chat UI
  fires overlapping requests; serialising at the type level is cheaper than
  auditing call sites. Generation runs on a detached task so the actor is never
  blocked for a whole turn.
- **Prefill and decode are timed separately** and both are shown under each
  reply. Folding them into one number would hide that HELIX accelerates prefill
  and deliberately does not touch decode.
- **`n_ubatch = 512`** keeps every SSM_SCAN call well above HELIX's 64-token
  gate, which is where its chunk-parallel path pays. A smaller ubatch would push
  calls below the gate and silently give up the speedup.
- **KV cache persists across turns** (see above). "New Chat" calls
  `llama_memory_clear` and drops the rendered history.
- **Stop actually stops.** The generation task is detached, so cancelling the
  consumer's task does not cancel it; the stream's `onTermination` sets an actor
  flag that the decode loop checks each iteration. Without it, Stop only stopped
  the UI from listening while the engine kept decoding to the token limit.
- **A system prompt is rendered on every turn** and kept out of the message
  history, so "New Chat" cannot drop it and multi-turn cannot duplicate it.
- **The C handles live in a small boxed class**, not directly on the actor. An
  actor's deinit cannot touch non-Sendable state; `isolated deinit` can, but it
  raises the deployment floor to macOS 15.4 *and* trips a "circular reference"
  compiler error under release whole-module optimization in Swift 6.3.3. Boxing
  sidesteps both without pushing cleanup onto callers.
- **`load_mode = MMAP`**: weights are mapped once and Metal maps the same pages
  under unified memory, so a 3.02 GiB model costs 3.68 GB resident with an 8192
  context rather than double that.

## Entitlements

`swift run` needs none — it is a plain executable reading a file the user owns.

For a distributed app bundle:

- **Sandbox off**, or add `com.apple.security.files.user-selected.read-only` and
  load the model through `NSOpenPanel`. A sandboxed app cannot read an arbitrary
  absolute path.
- **`com.apple.security.network.client` is not needed** — inference is entirely
  local.
- Hardened Runtime is fine as-is; no JIT or unsigned-memory entitlements are
  required, as Metal shaders are precompiled into the metallibs.

## iOS

The engine target is portable — it is plain Swift over llama.cpp's C API, with
no AppKit. Two changes are needed for an iOS target:

1. Replace `Sources/HelixChatUI/App.swift`; its `AppDelegate` exists only to
   promote a SwiftPM executable's activation policy, which an app bundle does
   not need. The `NSApplication` reference and `Color(nsColor:)` uses in
   `ChatView.swift` need `UIKit` equivalents.
2. Build llama.cpp and HELIX for `arm64-apple-ios` and link the static archives
   rather than the dylibs used here.

The 3.02 GiB model exceeds what most iPhones will let a single app hold
resident; this is a Mac-first target in practice.

## Development switches

| variable | effect |
|---|---|
| `HELIX_DEBUG_PROMPT=1` | dump the rendered chat prompt to stderr |
| `HELIX_UI_PREVIEW=1` | seed a fake conversation and skip model loading, for UI work |
| `HELIX_LLAMA_BUILD` | point the package at a different llama.cpp build |

## Repetition and the penalty window

Heavily quantized models loop. The useful finding here is *where*: measured over
a 4-turn conversation with 300-token replies, repetition **within** each reply is
0% in every configuration. Nothing loops internally. What happens is the model
restating its previous answers, and `penalty_last_n` decides whether the
sampler can even see them.

| `penalty_last_n` | whole-transcript repetition | worst repeated span |
|---|---|---|
| no penalty | 50.2% | 80 words |
| 64 (llama.cpp default) | 45.1% | 80 |
| 512 | 51.9% | 80 |
| **2048 (shipped)** | **17.3%** | **57** |

A 64-token window cannot reach past the current turn, so it is structurally
incapable of suppressing cross-turn repetition — which is why the llama.cpp
default is the wrong value for a chat client with long replies.

Raising the *strength* instead makes things worse (`penalty_repeat` 1.25 at
window 2048 scored 27.9% against 1.15's 17.3%), so the window is the lever.

Caveats worth keeping in mind: one 4-turn conversation per configuration, and
the probe prompt ("Continue with more detail." three times) actively invites
restatement, so 50% is a worst case rather than a typical one. The 64 -> 2048
effect is a 3x change and well clear of that noise; the 1.15 vs 1.25 comparison
is not.

`swift run helix-smoke` reproduces the table.

## Known gaps

- **Model quality is the ceiling, not the plumbing.** At 2.06 bpw the model
  follows direct instructions well but not meta-instructions: asked to "repeat
  my first message", it produces another carrot-cake answer. The rendered prompt
  is verifiably correct (`HELIX_DEBUG_PROMPT=1`), so this is quantization, not
  templating.
- KV reuse is partial in some turns — see above.
- No conversation persistence; "New Chat" discards history.
- Sampling parameters are not exposed in the UI (see `SamplingConfig`).

