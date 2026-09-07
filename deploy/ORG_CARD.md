---
title: README
emoji: "⚡"
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
---

# SG Systems

On-device inference for Apple Silicon. We work on the two constraints that
actually decide whether a model runs on a laptop or a phone:

**Capacity** — a 7B model at fp16 is 15.5 GB; Apple platforms terminate a
process near ~3.7 GB. Closing a 4x gap means deciding which weights can be
destroyed and which cannot.

**Bandwidth** — once the weights fit, throughput depends on whether the compute
maps onto the hardware at all. Sequential state-space recurrences do not map
onto matrix units by default, and mobile compilers fall back to scalar paths.

Those are separate problems, and solving one does nothing for the other.

## Projects

**FORGE** — mixed-precision post-training quantization. Not uniform low-bit:
the result depends less on the average bit width than on which tensor families
are excluded from it. Crushing a Mamba-2 state projection to ternary produces a
model that is fluent and confidently wrong.

**HELIX** — a Metal prefill kernel for the SSM scan in Mamba-2 hybrids,
integrated into `llama.cpp` through a 42-line patch. Chunk-parallel
decomposition; ~3x on the isolated op for the shapes upstream's fast path
declines.

## Models

| | |
|---|---|
| [`Falcon-H1-7B-FORGE-v2`](https://huggingface.co/sgsystems/Falcon-H1-7B-FORGE-v2) | `Falcon-H1-7B-Instruct` at 2.06 bpw / 3.48 GB. Runs on Apple Silicon in 3.73 GB resident at 75 tok/s decode. |

**Read the model card before using it.** This one needs `repeat_last_n = 2048`
— llama.cpp's default of 64 is too small to suppress the cross-turn repetition
these models fall into, and shipping the usual sampling triple without it
reproduces the bug.

## How we report numbers

Measurements here come with the conditions attached, and with the cases that
did not work.

Some things we published that were not what we first assumed:

- On M5, `simdgroup_matrix` at bf16 is **slower** than at fp32 (10.88 vs 16.12
  TFLOP/s). It is not the matrix hardware. Reduced precision only pays through
  `MetalPerformancePrimitives`, and only with both operands bf16.
- Adding a repetition penalty did **not** fix our repetition loops. The penalty
  *window* did, and only above 2048 tokens. Within-turn repetition was 0% the
  whole time; the failure was cross-turn.
- The quantized models still lose meta-instruction following. Direct
  instructions work; "repeat my first message" does not.

Where a claim is not measured, we say so rather than rounding it up.
