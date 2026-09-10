---
base_model: tiiuae/Falcon-H1-7B-Instruct
library_name: gguf
pipeline_tag: text-generation
license: other
license_name: falcon-llm-license
license_link: https://falconllm.tii.ae/falcon-terms-and-conditions.html
tags:
  - gguf
  - quantized
  - mamba2
  - hybrid
  - apple-silicon
  - llama.cpp
  - edge
language:
  - en
quantized_by: sgsystems
---

# Falcon-H1-7B-FORGE-v2

`Falcon-H1-7B-Instruct` quantized to **2.06 bpw average / 3.48 GB** with FORGE,
a mixed-precision post-training method. **3.73 GB resident at 75 tok/s decode on
an M5 Max**; runs on an iPhone (A16, 6 GB) at 5.5 tok/s decode — see the memory
envelope and open issues below before planning a phone deployment.

Hybrid Mamba-2 + attention, 7.59B parameters, 44 blocks.

| | |
|---|---|
| File | `falcon-h1-7b-forge-v2.gguf` (3.48 GB) |
| Format | GGUF v3, 751 tensors |
| Parameters | 7.59 B |
| Architecture | `falcon-h1` — 44 blocks, hybrid Mamba-2 + attention |
| SSM | `d_state` 256, `d_inner` 3072, 24 heads, 1 group |
| Vocab | 130 049 |
| Trained context | 262 144 (use 4096 here — see memory envelope) |
| BOS / EOS | 17 / 11 |
| Chat template | ChatML, embedded in the GGUF |

### Quantization layout

Read straight from the file — this is the design decision, made concrete:

| tensor family | count | type |
|---|---|---|
| `ssm_out` | 44 | **Q6_K** |
| `ffn_down` | 44 | **Q6_K** |
| `output` | 1 | Q6_K |
| `token_embd` | 1 | Q4_K |
| `attn_q` / `attn_k` / `attn_v` / `attn_output` | 176 | `TQ2_0` |
| `ffn_gate` / `ffn_up` | 88 | `TQ2_0` |
| `ssm_in` | 44 | `TQ2_0` |
| norms, `ssm_conv1d`, `ssm_dt` | 353 | F32 |

308 of 751 tensors are ternary; the 89 held at Q6_K are what the model needs to
stay factual.

## Memory envelope

**The 3.48 GB file size is not the memory requirement.** Peak resident set,
measured on an M5 Max with `n_ubatch = 512`:

| `n_ctx` | peak RSS |
|---|---|
| 2048 | 3.59 GB |
| **4096 (recommended)** | **3.68 GB** |
| 8192 | 3.86 GB |

**Use `n_ctx = 4096`** on a Mac; 8192 costs another ~180 MB of KV cache.
A native SwiftUI host adds roughly 50 MB on top.

### On iOS the binding constraint is different

Measured on an iPhone 14 Pro Max (A16, 6 GB), the process **footprint peaks at
0.31 GB** — the mmap'd weights are clean, file-backed pages and are not charged
to `phys_footprint`, so the jetsam limit is never approached. What binds instead
is Metal's `recommendedMaxWorkingSetSize` (4096 MiB on that device):

| | MiB |
|---|---|
| model weight buffer views | 3626.47 |
| KV cache (`n_ctx` 2048) | 88.00 |
| recurrent SSM state | 133.80 |
| compute buffer (`n_ubatch` 256) | 170.01 |
| **total** | **4018.28** of 4096.02 |

`n_ubatch = 256` is required on 6 GB devices — at 512 the compute buffer is
340 MiB, which puts the total 92 MiB over and every command buffer fails with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. Note that
`com.apple.developer.kernel.increased-memory-limit` raises the *jetsam* limit and
has no effect on this ceiling.

Load with `mmap` so the OS and Metal share the same pages under unified memory.

## Quality

Wikitext2 perplexity, measured against the F16 source:

| build | size | wikitext2 ppl | vs F16 | factual probes | decode t/s |
|---|---:|---:|---:|---:|---:|
| F16 | 15.18 GB | 6.5875 | 1.00x | 12/12 | 29.1 |
| v1 (`ssm_out` ternary, 32 calib) | 3.25 GB | 10.7590 | 1.63x | 9/12 | ~63 |
| **v2 (`ssm_out` Q6_K, 128 calib)** | **3.48 GB** | **9.1633** | **1.39x** | **12/12** | ~63 |

+0.23 GB over v1 buys back full factual recall on the probe set and 15% of the
perplexity gap, at 4.4x smaller than F16.

What v1 got wrong is worth seeing, because two of the three failures were not
loops — they were fluent and confidently incorrect:

| probe | v1 | v2 |
|---|---|---|
| carrot cake ingredients | `"carrot cake mix, carrot cake mix, …"` | `"Carrots, sugar, flour, eggs, butter, cinnamon…"` |
| capital of Australia | **`"Sydney"`** | `"Canberra"` |
| boiling point of water | **`"99°C"`** | `"100°C"` |

**Read this with two caveats.** The v2 run changed two variables at once —
preserving `ssm_out` *and* raising calibration from 32 to 128 sequences — so the
recovery cannot be attributed between them; the Hessian flatness outlier on
`ssm_out` (6.78 against ~0.33 elsewhere) makes it the likelier cause, but that is
inference. And there is no same-size `Q4_K_M` baseline, so "better than
conventional 4-bit at this footprint" is **not** a claim this data supports.

## Required sampling parameters

This model **will** produce structural repetition without these. They are not
suggestions.

| parameter | value |
|---|---|
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `top_k` | 40 |
| `repeat_penalty` | 1.15 |
| **`repeat_last_n`** | **2048** |

```sh
llama-cli -m falcon-h1-7b-forge-v2.gguf -c 4096 -ub 512 \
  --temp 0.7 --top-p 0.9 --top-k 40 \
  --repeat-penalty 1.15 --repeat-last-n 2048
```

<details>
<summary>llama-cpp-python</summary>

```python
from llama_cpp import Llama

llm = Llama(
    model_path="falcon-h1-7b-forge-v2.gguf",
    n_ctx=4096, n_ubatch=512, n_gpu_layers=-1, use_mmap=True,
)
out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful, knowledgeable, and precise AI assistant."},
        {"role": "user", "content": "Make me a carrot cake."},
    ],
    temperature=0.7, top_p=0.9, top_k=40,
    repeat_penalty=1.15,          # and set repeat_last_n=2048 on the context
)
print(out["choices"][0]["message"]["content"])
```

</details>

### Why `repeat_last_n` matters more than `repeat_penalty`

llama.cpp defaults `repeat_last_n` to 64, which is **the wrong value for this
model**. Measured over a 4-turn conversation with 300-token replies, repetition
*within* each individual reply is 0% in every configuration — nothing loops
internally. The failure is the model restating its previous answers, and a
64-token window cannot reach back far enough to penalize them:

| `repeat_last_n` | whole-transcript repetition | worst verbatim span |
|---|---|---|
| penalty disabled | 50.2% | 80 words |
| 64 (llama.cpp default) | 45.1% | 80 |
| 512 | 51.9% | 80 |
| **2048** | **17.3%** | **57** |

Raising the penalty *strength* instead makes it worse (1.25 at window 2048
scored 27.9%). Set the window.

**This is reduced, not solved.** Expect some cross-turn restatement in long
conversations.

## Chat template

ChatML, embedded in the GGUF — `llama_chat_apply_template` picks it up with no
extra configuration. Applying it is required: without it the model continues text
instead of answering.

```
<|im_start|>system
You are a helpful, knowledgeable, and precise AI assistant.<|im_end|>
<|im_start|>user
Make me a carrot cake.<|im_end|>
<|im_start|>assistant
```

A system prompt measurably helps instruction adherence at this bit width.

## How it was quantized

FORGE runs sequential GPTQ with a ternary solver, but **excludes two tensor
families from ternarization**, keeping them at Q6_K:

- `ssm_out` — the Mamba-2 state projection
- `ffn_down` — the FFN down-projection

That exclusion is the entire result. A uniformly-ternary build (v1) is fluent and
confidently wrong: it listed **yeast and rising dough as carrot cake
ingredients**. Carrot cake is chemically leavened, so this is not a rounding
error — the model had lost the association between a recipe name and its method.

| | v1 (uniform ternary) | v2 (this model) |
|---|---|---|
| size | 3.25 GB | 3.48 GB |
| carrot cake, single turn | 3.0% repetition, span 7 | **0.0%, span 3** |
| 4 accumulated turns | 48.1% repetition, span 80 | **19.7%, span 43** |

The 230 MB those tensors cost is the difference between a model that knows things
and one that does not.

Calibration: wikitext2 train, 128 samples at seqlen 2048, seed 0. Full
configuration is in the `.forge.json` sidecar shipped alongside the GGUF, so a
build is reproducible from the artifact.

## Performance

M5 Max, llama.cpp with the HELIX Metal prefill kernel:

| | HELIX off | HELIX on |
|---|---|---|
| prefill (2048 tok) | 1428.6 t/s | **1941.8 t/s** |
| prefill (8192 tok) | 1260.5 t/s | 1623.9 t/s |
| decode | 74.4 t/s | 74.6 t/s |

HELIX accelerates prefill only; decode is bandwidth-bound and untouched. The
model runs correctly on stock `llama.cpp` — HELIX is an optional speedup, not a
requirement.

## Intended use

Built for **on-device assistant workloads on Apple Silicon** where the memory
budget is the binding constraint — a 7B-class model that fits beside a running
OS rather than one that needs a workstation.

Not intended for: batch serving (the memory savings buy nothing when VRAM is
plentiful and the quantization costs accuracy), tasks needing long multi-turn
coherence (see below), or anything where a factual error is expensive.

## Behavioural limitations

- **Cross-turn repetition persists** at ~19.7% over four turns even with correct
  sampling.
- **Meta-instructions fail.** Direct instructions ("list ingredients only, no
  steps") are followed; instructions about the conversation ("repeat my first
  message") are not. This is a quantization ceiling.
- **English only**, inherited from the calibration set and evaluation.
- **Not evaluated on standard benchmarks.** No MMLU or HellaSwag. Perplexity
  against the fp16 source *has* been measured (see Quality above); a same-size
  `Q4_K_M` baseline has not. Treat this as an engineering artifact, not a
  leaderboard entry.
- Inherits all limitations and the license of `tiiuae/Falcon-H1-7B-Instruct`.

## Known Limitations & Open Issues

**#1 — No same-size baseline.** Perplexity against F16 is measured (1.39x, above),
but a standard `Q4_K_M` build has not been. Whether 2.06 bpw mixed-precision
actually beats conventional 4-bit at comparable size is open. The v2 attribution
is also confounded (see Quality).

**#2 — Overlapping buffer views waste 312 MiB.** The mapped model is 3313.51 MiB,
but exceeding Metal's `maxBufferLength` makes ggml split it into two overlapping
views (3072.00 + 554.09 MiB) that both count against the GPU working set. On a
6 GB iPhone this leaves the app at 98% of its budget, where prefill completes but
is intermittently `SIGKILL`ed.

**#3 — Decode gates mobile UX.** 5.5 tok/s on A16 — a 146-token reply takes 26 s.
That is ~18 GB/s effective, roughly half the chip's LPDDR5 ceiling, so it is
bandwidth-bound rather than broken. HELIX accelerates prefill only, by design; a
fused single-step decode kernel is what interactive phone speed requires.

**#4 — The HELIX MPP path has never engaged for this model.** `HELIX_MPP_K` is a
compile-time 128 and Falcon-H1 uses `d_state = 256`, so the runtime falls back
silently to the fp32 `simdgroup_matrix` path. The 4.62x MPP figure quoted
elsewhere for HELIX has never applied to this model on any hardware. Supporting
`d_state = 256` needs a second descriptor instantiation or `dynamic_extent`.

## Files

| file | |
|---|---|
| `falcon-h1-7b-forge-v2.gguf` | the model, 3.48 GB |
| `falcon-h1-7b-forge-v2.forge.json` | quantization configuration sidecar |

## Links

| | |
|---|---|
| Overview | [github.com/sg-systems-co/forge-helix](https://github.com/sg-systems-co/forge-helix) |
| Quantization (FORGE) | [github.com/sg-systems-co/forge](https://github.com/sg-systems-co/forge) |
| Metal kernel (HELIX) | [github.com/sg-systems-co/helix](https://github.com/sg-systems-co/helix) |
| SwiftUI client | [github.com/sg-systems-co/helix-chat-ui](https://github.com/sg-systems-co/helix-chat-ui) |
