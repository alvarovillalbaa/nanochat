"""
Generate preference pairs for DPO training.
This script uses a teacher model to generate multiple candidate responses
and then ranks them to create (chosen, rejected) pairs.

Usage:
    python -m scripts.gen_preference_data --output-file preference_data.jsonl --num-examples 1000
"""

import os
import json
import argparse
from typing import List, Dict
import random

from nanochat.common import get_base_dir

# Import prompt generators from gen_teacher_data
from scripts.gen_teacher_data import (
    OPENAI_AVAILABLE,
    generate_general_chat_prompt,
    generate_code_prompt,
    generate_math_prompt,
    generate_reasoning_prompt,
    TASK_MIXTURE,
    create_openai_client,
)

# -----------------------------------------------------------------------------


def generate_candidate_responses(
    prompt: str,
    num_candidates: int = 3,
    teacher_model: str = "gpt-4o-mini",
    temperature: float = 0.9,
    client=None,
) -> List[str]:
    """
    Generate multiple candidate responses for a given prompt.
    Higher temperature encourages diversity.
    """
    if client is None and not OPENAI_AVAILABLE:
        raise RuntimeError(
            "The OpenAI SDK is required: install it with `pip install openai`"
        )
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")
    if client is None:
        client = create_openai_client()
    candidates = []

    for _ in range(num_candidates):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = client.chat.completions.create(
                model=teacher_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
            )
        except Exception as exc:
            raise RuntimeError("Candidate-generation request failed") from exc
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Candidate-generation model returned an empty response")
        candidates.append(content)

    return candidates


def rank_responses(
    prompt: str,
    candidates: List[str],
    teacher_model: str = "gpt-4o-mini",
    client=None,
) -> List[int]:
    """
    Use the teacher model to rank the candidate responses.
    Returns indices sorted from best to worst.

    For a more robust implementation, you could:
    - Ask the model to score each response
    - Use pairwise comparisons
    - Use a separate reward model
    """
    if client is None and not OPENAI_AVAILABLE:
        raise RuntimeError(
            "The OpenAI SDK is required: install it with `pip install openai`"
        )
    if len(candidates) <= 1:
        raise ValueError("At least two candidates are required for ranking")
    if client is None:
        client = create_openai_client()

    # Create a prompt asking the model to rank the responses
    ranking_prompt = f"""Given the user prompt: "{prompt}"

Please rank the following {len(candidates)} assistant responses from best to worst based on:
- Accuracy and correctness
- Helpfulness and completeness
- Clarity and conciseness
- Following instructions

Responses:
"""

    for i, candidate in enumerate(candidates):
        ranking_prompt += f"\n{i + 1}. {candidate}\n"

    ranking_prompt += "\nRespond with ONLY the ranking as numbers separated by commas, e.g., '2,1,3' means response 2 is best, then 1, then 3."

    messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator of AI assistant responses.",
        },
        {"role": "user", "content": ranking_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=teacher_model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent rankings
            max_tokens=50,
        )
    except Exception as exc:
        raise RuntimeError("Preference-ranking request failed") from exc

    # Parse and validate a complete permutation; never invent random labels.
    ranking_content = response.choices[0].message.content
    if not isinstance(ranking_content, str) or not ranking_content.strip():
        raise RuntimeError("Preference-ranking model returned an empty response")
    ranking_str = ranking_content.strip()
    ranking_str = "".join(c for c in ranking_str if c.isdigit() or c == ",")
    ranking = [int(x.strip()) - 1 for x in ranking_str.split(",") if x.strip()]
    if len(ranking) != len(candidates) or set(ranking) != set(range(len(candidates))):
        raise RuntimeError(
            f"Preference-ranking model returned an invalid permutation: {ranking_str!r}"
        )
    return ranking


def generate_preference_pair(
    task_type: str,
    teacher_model: str = "gpt-4o-mini",
    num_candidates: int = 3,
    client=None,
) -> Dict:
    """
    Generate a single preference pair (prompt, chosen, rejected).
    """
    # Generate prompt based on task type
    if task_type == "general_chat":
        prompt = generate_general_chat_prompt()
    elif task_type == "code":
        prompt = generate_code_prompt()
    elif task_type == "math":
        prompt = generate_math_prompt()
    elif task_type == "reasoning":
        prompt = generate_reasoning_prompt()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Generate multiple candidate responses
    candidates = generate_candidate_responses(
        prompt,
        num_candidates,
        teacher_model,
        client=client,
    )

    # Rank the candidates
    ranking = rank_responses(prompt, candidates, teacher_model, client=client)

    # Create preference pair: best vs worst (or best vs second-best for harder training)
    best_idx = ranking[0]
    worst_idx = ranking[-1]  # Use worst for clearer signal
    # worst_idx = ranking[1]  # Alternatively, use second-best for harder discrimination

    return {
        "prompt": prompt,
        "chosen": candidates[best_idx],
        "rejected": candidates[worst_idx],
    }


def generate_preference_dataset(
    num_examples: int,
    output_file: str,
    task_mixture: Dict[str, float] = TASK_MIXTURE,
    teacher_model: str = "gpt-4o-mini",
    num_candidates: int = 3,
    resume: bool = True,
    client=None,
):
    """
    Generate a full dataset of preference pairs.
    """
    # Determine starting index if resuming
    start_idx = 0
    if resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            start_idx = sum(1 for _ in f)
        print(f"Resuming from example {start_idx}")
        output_mode = "a"
    else:
        output_mode = "w"

    # Create task distribution
    task_types = list(task_mixture.keys())
    task_weights = list(task_mixture.values())

    # Generate examples
    with open(output_file, output_mode) as f:
        for i in range(start_idx, num_examples):
            # Sample task type
            task_type = random.choices(task_types, weights=task_weights)[0]

            pair = generate_preference_pair(
                task_type,
                teacher_model,
                num_candidates,
                client=client,
            )

            f.write(json.dumps(pair) + "\n")
            f.flush()

            if (i + 1) % 10 == 0:
                print(
                    f"Generated {i + 1}/{num_examples} preference pairs (task: {task_type})"
                )

    print("\n✅ Preference dataset generation complete!")
    print(f"   Saved to: {output_file}")
    print(f"   Total examples: {num_examples}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate preference pairs for DPO training"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="preference_data.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=500,
        help="Number of preference pairs to generate",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gpt-4o-mini",
        help="Teacher model for generation and ranking",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of candidate responses to generate per prompt",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from existing file"
    )

    args = parser.parse_args()

    client = create_openai_client()

    # Ensure output directory exists
    base_dir = get_base_dir()
    output_file = os.path.join(base_dir, args.output_file)
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    # Generate dataset
    generate_preference_dataset(
        num_examples=args.num_examples,
        output_file=output_file,
        teacher_model=args.teacher_model,
        num_candidates=args.num_candidates,
        resume=not args.no_resume,
        client=client,
    )
