# nanochat pipeline overview

nanochat is a compact, end-to-end language-model project. Its stages share a
tokenizer and pass checkpoints through a fixed sequence:

1. Train the tokenizer.
2. Pretrain a base language model.
3. Midtrain on chat, multiple-choice, math, and calculator-use patterns.
4. Supervised-fine-tune the chat model.
5. Optionally run GSM8K reinforcement learning or offline DPO.
6. Evaluate and serve a selected checkpoint.
7. Assemble the run report.

## Tokenizer

`scripts/tok_train.py` trains the custom Rust BPE implementation and exports an
inference tokenizer. The vocabulary must stay fixed for every checkpoint that
uses it. See [`tokenization.md`](tokenization.md).

## Base pretraining

`scripts/base_train.py` trains next-token prediction on locally downloaded
FineWeb-Edu parquet shards. It tracks validation bits per byte and the
project-defined CORE metric. See [`data.md`](data.md) and
[`training.md`](training.md).

## Midtraining

`scripts/mid_train.py` combines SmolTalk, MMLU auxiliary data, and GSM8K. This
stage teaches the conversation token format, multiple-choice responses, and
the calculator token pattern used by the inference engine.

## Chat post-training

`scripts/chat_sft.py` performs supervised chat training on ARC, GSM8K,
SmolTalk, and optionally a validated custom teacher JSONL dataset.

Two optional follow-up paths are available:

- `scripts/chat_rl.py` uses a deliberately simplified, REINFORCE-like quoted
  "GRPO" loop with verifiable GSM8K rewards.
- `scripts/chat_dpo.py` optimizes validated chosen/rejected pairs relative to a
  frozen SFT reference model.

## Evaluation

`scripts/chat_eval.py` reports ARC-Easy, ARC-Challenge, MMLU, GSM8K, and
HumanEval accuracy. When all five run, it also computes ChatCORE, a
random-baseline-centered mean. Per-task results remain the primary evidence.

HumanEval executes generated programs through a best-effort subprocess
reliability guard. It is not a security sandbox; see [`sandbox.md`](sandbox.md).

## Inference and interfaces

`nanochat/engine.py` implements prefill/decode generation and KV caching. Its
special Python tokens currently support restricted numeric calculator
expressions, not arbitrary code execution. `scripts/chat_cli.py` and
`scripts/chat_web.py` expose the engine for local research use. See
[`inference.md`](inference.md).

## Evidence boundary

The repository's Markdown report summarizes recorded runs. Architecture or
training changes should not be described as improvements until matched
training and evaluation demonstrate the effect.
