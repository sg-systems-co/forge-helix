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
---

# Falcon-H1-7B-FORGE-v2

`Falcon-H1-7B-Instruct` quantized to **2.06 bpw average / 3.48 GB** with FORGE,
a mixed-precision post-training method. Runs on Apple Silicon in **3.73 GB
resident** at **75 tok/s** decode.

Hybrid Mamba-2 + attention, 7.59B parameters, 44 blocks.

| | |
|---|---|
| File | `falcon-h1-7b-forge-v2.gguf` (3.48 GB) |
| Format | GGUF, `TQ2_0` ternary with Q6_K exclusions |
| Context | 4096 recommended (see memory envelope) |
| Chat template | ChatML, embedded in the GGUF |

## Memory envelope

**The 3.48 GB file size is not the memory requirement.** Peak resident set,
measured on an M5 Max with `n_ubatch = 512`:

| `n_ctx` | peak RSS |
|---|---|
| 2048 | 3.59 GB |
| **4096 (recommended)** | **3.68 GB** |
| 8192 | 3.86 GB |

Apple platforms terminate a process near ~3.7 GB. **Use `n_ctx = 4096`**;
8192 exceeds the budget on memory-constrained devices. A native SwiftUI host adds
roughly 50 MB on top.

Load with `mmap` so the OS and Metal share the same pages under unified memory.

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

## Limitations

- **Cross-turn repetition persists** at ~19.7% over four turns even with correct
  sampling.
- **Meta-instructions fail.** Direct instructions ("list ingredients only, no
  steps") are followed; instructions about the conversation ("repeat my first
  message") are not. This is a quantization ceiling.
- **English only**, inherited from the calibration set and evaluation.
- **Not evaluated on standard benchmarks.** Claims here rest on targeted factual
  probes and repetition measurement, not MMLU/HellaSwag. Treat it as an
  engineering artifact, not a leaderboard entry.
- Inherits all limitations and the license of `tiiuae/Falcon-H1-7B-Instruct`.

## Files

| file | |
|---|---|
| `falcon-h1-7b-forge-v2.gguf` | the model, 3.48 GB |
| `falcon-h1-7b-forge-v2.forge.json` | quantization configuration sidecar |

## Links

FORGE, HELIX and the SwiftUI client: https://github.com/sgsystems/forge-helix
