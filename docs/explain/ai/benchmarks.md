# Evaluation Benchmarks

The `nanochat` project integrates several standard benchmarks to evaluate model performance across different capabilities (reasoning, knowledge, math, coding). These are implemented in the `tasks/` directory.

## 1. ARC (AI2 Reasoning Challenge)
*   **Source**: [Allen Institute for AI](https://allenai.org/data/arc)
*   **Implementation**: `tasks/arc.py`
*   **Type**: Multiple Choice (Categorical)
*   **Subsets**:
    *   `ARC-Easy`: Grade-school science questions.
    *   `ARC-Challenge`: Harder questions requiring reasoning.
*   **Evaluation Method**: The model is presented with the question and choices. We check if the generated answer matches the correct letter (A, B, C, D).

## 2. GSM8K (Grade School Math 8K)
*   **Source**: [OpenAI](https://github.com/openai/grade-school-math)
*   **Implementation**: `tasks/gsm8k.py`
*   **Type**: Generative (Chain of Thought / Tool Use)
*   **Content**: High quality grade school math word problems.
*   **Evaluation Method**:
    *   The model generates a solution, potentially using the calculator tool.
    *   The final answer is expected to be formatted with `#### <number>`.
    *   We extract this number and compare it to the ground truth.
*   **Role in RL**: This task is used as the primary environment for Reinforcement Learning (RL), where the reward is simply `1.0` if the answer is correct and `0.0` otherwise.

## 3. MMLU (Massive Multitask Language Understanding)
*   **Source**: [Hendrycks et al.](https://arxiv.org/abs/2009.03300)
*   **Implementation**: `tasks/mmlu.py`
*   **Type**: Multiple Choice
*   **Content**: Covers 57 subjects across STEM, the humanities, the social sciences, and more.
*   **Goal**: Measure world knowledge and problem solving.

## 4. HumanEval
*   **Source**: [OpenAI](https://github.com/openai/human-eval)
*   **Implementation**: `tasks/humaneval.py`
*   **Type**: Code Generation
*   **Content**: Python coding problems.
*   **Evaluation Method**:
    *   The model generates a function body.
    *   The code is executed against unit tests in a child process with
        reliability guards. This is not a security sandbox; see
        `docs/explain/ai/sandbox.md`.
    *   Metric: `pass@k` (percentage of problems solved).

## 5. SmolTalk
*   **Source**: [HuggingFaceTB](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
*   **Implementation**: `tasks/smoltalk.py`
*   **Type**: Conversational / Instruction Following
*   **Goal**: Used primarily for **Midtraining** to teach the model the chat format and tool-use patterns.
