# Experimental quality roadmap

This roadmap records research directions for improving nanochat. It is not an
implementation or performance claim. Current status and verification are kept
in [`../IMPLEMENTATION_CHECKLIST.md`](../IMPLEMENTATION_CHECKLIST.md).

## Guiding constraint

Preserve nanochat's compact, understandable pipeline. Add experiments behind
explicit configuration, keep original defaults compatible, and require
matched-run evidence before adopting a new default.

## 1. Data quality and distillation

### Teacher-generated SFT

Generate diverse prompts across general chat, code, math, and reasoning, then
request complete responses from a configured teacher provider. Store only
validated JSONL conversations and review a representative sample before
training.

Implementation requirements:

- fail before writing when provider configuration or requests fail;
- never substitute placeholder/error text for a teacher response;
- record model and generation settings with the dataset artifact;
- preserve a held-out evaluation split and provenance.

### Preference data

Generate multiple responses per prompt, rank them, and retain explicit
chosen/rejected pairs. Rankings must be complete permutations; invalid provider
output must stop generation rather than create random labels.

### Base-data experiments

Potential experiments include deduplication, quality filtering, and scheduled
domain mixtures. Each requires data-provenance, contamination, throughput, and
matched-evaluation evidence before inclusion.

## 2. Post-training

### DPO

Use offline preference pairs with a frozen copy of the starting SFT model.
Aggregate log probabilities over supervised assistant tokens and optimize the
reference-relative DPO objective. Validate a small training run, checkpoint
reload, loss behavior, and held-out preference metrics.

### Tool-use training

The current inference engine supports restricted calculator expressions. A
broader tool-use experiment would require:

- an explicit schema for calls and outputs;
- deterministic execution traces;
- an actual security boundary for untrusted code;
- success and failure evaluations, including malformed calls.

## 3. Architecture experiments

### SwiGLU

SwiGLU is available only as an opt-in architecture for new base checkpoints.
It changes parameter names and shapes, so it is not a runtime toggle for legacy
checkpoints. Compare parameter count, throughput, loss, and downstream metrics
against the ReLU-squared baseline before considering a default change.

### Longer context and attention

The model already uses rotary embeddings and PyTorch scaled dot-product
attention. Longer-context work still needs training data, memory measurements,
and retrieval-style evaluation; changing a sequence-length value alone does
not prove useful context extension.

### Sparse experts

Mixture-of-experts layers would materially increase implementation and routing
complexity. Treat them as a separate research branch with capacity, load
balancing, communication, and quality evidence.

## 4. Training efficiency

Candidate experiments include alternative schedules, gradient checkpointing,
and compute/loss curve fitting. Keep them configuration-driven and compare
tokens, FLOPs, wall time, memory, and evaluation at matched budgets.

## 5. Evaluation gates

Every proposed improvement should pass:

1. focused unit and checkpoint-compatibility tests;
2. a small end-to-end GPU smoke run;
3. matched baseline and experiment training;
4. per-task ARC, MMLU, GSM8K, and HumanEval reporting as relevant;
5. held-out data/preference checks for post-training changes;
6. throughput, memory, and cost reporting;
7. a review for data provenance, leakage, and unsafe execution paths.

Composite scores can summarize results, but they must not replace per-task
metrics or the underlying run configuration.
