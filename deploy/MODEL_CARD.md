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
a mixed-precision post-training method. Runs on Apple Silicon in **3.73 GB
resident** at **75 tok/s** decode.

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

## Limitations

- **Cross-turn repetition persists** at ~19.7% over four turns even with correct
  sampling.
- **Meta-instructions fail.** Direct instructions ("list ingredients only, no
  steps") are followed; instructions about the conversation ("repeat my first
  message") are not. This is a quantization ceiling.
- **English only**, inherited from the calibration set and evaluation.
- **Not evaluated on standard benchmarks.** Everything quoted here comes from
  targeted factual probes and automated repetition measurement — no MMLU,
  HellaSwag or perplexity-vs-baseline numbers. Treat it as an engineering
  artifact, not a leaderboard entry. In particular, **no perplexity comparison
  against the fp16 source or a standard Q4_K_M build has been run**, so the
  accuracy cost of 2.06 bpw is characterised only where it was probed.
- Inherits all limitations and the license of `tiiuae/Falcon-H1-7B-Instruct`.

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
