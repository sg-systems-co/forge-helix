---
title: README
emoji: "⚡"
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
---

# SG Systems

**Consulting & forward-deployed engineering for Southeast Asian applied AI.**

We embed senior engineers with client teams and ship working systems in weeks
rather than producing strategy decks. Fixed scope, outcome-based pricing, no
hourly billing.

Serving banking, insurance, fintech, healthcare, retail, government and
mid-market commerce across SEA — where a deployment has to be PDPA-compliant,
locally tuned, and native to the channels people actually use.

[sg-systems.co](https://www.sg-systems.co/)

## Why we publish inference work

Regional deployment runs into two constraints that a general-purpose API does
not solve.

**Data residency.** PDPA and its neighbours make "send the customer's data to a
US endpoint" a compliance conversation before it is an engineering one.
Inference that never leaves the device sidesteps that conversation entirely.

**Hardware reality.** SEA deployments run on the devices people have, not on
datacentre accelerators. A model that needs 15 GB is not a product; one that
fits in 3.7 GB is.

So we work on making capable models run locally. What we publish here are the
artifacts and measurements from that work.

## Projects

**FORGE** — mixed-precision post-training quantization. The result depends less
on the average bit width than on which tensor families are excluded from it:
crushing a Mamba-2 state projection to ternary yields a model that is fluent and
confidently wrong.

**HELIX** — a Metal prefill kernel for the SSM scan in Mamba-2 hybrids,
integrated into `llama.cpp` through a 42-line patch. Chunk-parallel
decomposition; ~3x on the isolated op for shapes upstream's fast path declines.

Source: [github.com/sg-systems-co/forge-helix](https://github.com/sg-systems-co/forge-helix)

## Models

| | |
|---|---|
| [`Falcon-H1-7B-FORGE-v2`](https://huggingface.co/sgsystems/Falcon-H1-7B-FORGE-v2) | `Falcon-H1-7B-Instruct` at 2.06 bpw / 3.48 GB. Runs on Apple Silicon in 3.73 GB resident at 75 tok/s decode. **English only** — this is an engineering artifact from the compression work, not a SEA-language model. |

**Read the model card before using it.** It needs `repeat_last_n = 2048`;
llama.cpp's default of 64 is too small to suppress the cross-turn repetition
these models fall into, and the usual sampling triple without it reproduces the
bug.

## How we report numbers

Measurements come with their conditions attached, and with the cases that did
not work. Things we published that contradicted our own assumptions:

- On M5, `simdgroup_matrix` at bf16 is **slower** than at fp32 (10.88 vs 16.12
  TFLOP/s). It is not the matrix hardware. Reduced precision only pays through
  `MetalPerformancePrimitives`, and only with both operands bf16.
- Adding a repetition penalty did **not** fix our repetition loops. The penalty
  *window* did, and only above 2048 tokens. Within-turn repetition was 0%
  throughout; the failure was entirely cross-turn.
- The quantized model still loses meta-instruction following, and no perplexity
  comparison against fp16 has been run.

Where a claim is not measured, we say so rather than rounding it up.

## Contact

sebastian@sg-systems.co
