# Training Pipeline

The `nanochat` training pipeline is broken down into distinct stages, each handled by a specific script in the `scripts/` directory. This modular approach allows for flexible experimentation and checkpointing.

## Pipeline Stages

### 1. Tokenizer Training
*   **Script**: `scripts/tok_train.py`
*   **Goal**: Train the custom BPE tokenizer on a subset of the data.
*   **Output**: `tokenizer/` under nanochat's configured base directory.

### 2. Pretraining (Base Model)
*   **Script**: `scripts/base_train.py`
*   **Goal**: Train the model from scratch on the **FineWeb-Edu** dataset.
*   **Objective**: Next-token prediction on raw text.
*   **Key Features**:
    *   Uses `Muon` optimizer for 2D params, `AdamW` for others.
    *   Distributed training support.
    *   Saves checkpoints to `base_checkpoints/` under nanochat's configured base directory.

### 3. Midtraining (Instruction Adaptation)
*   **Script**: `scripts/mid_train.py`
*   **Goal**: Adapt the base model to conversational formats and tool use.
*   **Data**: **SmolTalk** dataset (conversations) + synthetic data.
*   **Objective**: Learn the chat format (`<|user_start|>`, etc.) and when to call tools.

### 4. Supervised Fine-Tuning (SFT)
*   **Script**: `scripts/chat_sft.py`
*   **Goal**: Supervise chat responses on the configured task mixture.
*   **Data**: ARC, GSM8K, SmolTalk, and optionally validated custom teacher JSONL.
*   **Evaluation**: Tracks SmolTalk validation loss and periodically runs MMLU and ARC-Easy.

### 5. Reinforcement Learning (RL)
*   **Script**: `scripts/chat_rl.py`
*   **Goal**: Further optimize specific capabilities (like math reasoning) using reinforcement learning.
*   **Method**: A deliberately simplified, quoted "GRPO" loop: on-policy
    REINFORCE-like updates without a PPO ratio/clip or reference-model KL.
*   **Reward Signal**: Verifiable correctness on tasks like **GSM8K** (math problems).

## Evaluation

*   **Script**: `scripts/chat_eval.py`
*   **Goal**: Run a suite of benchmarks to measure model performance.
*   **Benchmarks**:
    *   **ARC-Easy / ARC-Challenge**: Reasoning.
    *   **MMLU**: General knowledge.
    *   **GSM8K**: Math.
    *   **HumanEval**: Coding.
*   **Metric**: Per-task accuracy plus ChatCORE, a random-baseline-centered mean
    when all five chat tasks are evaluated.

## Serving

*   **Script**: `scripts/chat_web.py`
*   **Goal**: Serve the final model via a Web UI and API.
