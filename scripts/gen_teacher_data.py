"""
Generate teacher-distilled SFT data using a frontier model (e.g., OpenAI GPT models).
This script creates high-quality instruction-following data by:
1. Sampling prompts from a task mixture
2. Using a teacher model to generate responses (with optional CoT)
3. Saving the data in a format compatible with tasks/customjson.py

Usage:
    python -m scripts.gen_teacher_data --output-file data/teacher_sft.jsonl --num-examples 10000
"""

import os
import json
import argparse
from typing import Dict
import random

from nanochat.common import get_base_dir

# Optional: OpenAI client (requires OPENAI_API_KEY environment variable)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENAI_AVAILABLE = OpenAI is not None


def create_openai_client():
    """Create the optional OpenAI client or fail before generating training data."""
    if not OPENAI_AVAILABLE or OpenAI is None:
        raise RuntimeError(
            "The OpenAI SDK is required: install it with `pip install openai`"
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for teacher-data generation")
    return OpenAI(api_key=api_key)


# -----------------------------------------------------------------------------
# Task mixture configuration
# Adjust these weights to control the distribution of generated data
TASK_MIXTURE = {
    "general_chat": 0.40,  # General conversation and instruction-following
    "code": 0.20,  # Code generation and debugging
    "math": 0.20,  # Math word problems and reasoning
    "reasoning": 0.20,  # Logical reasoning and analysis
}

# -----------------------------------------------------------------------------
# Prompt templates for different task types

GENERAL_CHAT_PROMPTS = [
    "Write a professional email to a colleague about {topic}.",
    "Explain {concept} in simple terms for a beginner.",
    "What are the key differences between {item1} and {item2}?",
    "Provide 5 tips for {activity}.",
    "Summarize the main points about {topic}.",
    "Write a step-by-step guide on how to {task}.",
]

CODE_PROMPTS = [
    "Write a Python function that {task}.",
    "Debug this code: {buggy_code}",
    "Implement {algorithm} in {language}.",
    "Write unit tests for a function that {functionality}.",
    "Optimize this code for better performance: {code}.",
    "Explain this code snippet: {code}",
]

MATH_PROMPTS = [
    "Solve this word problem: {problem}",
    "If {condition}, calculate {query}.",
    "A {scenario} has {initial_state}. After {change}, what is {final_query}?",
    "Find the value of x when {equation}.",
    "Calculate the percentage increase from {value1} to {value2}.",
]

REASONING_PROMPTS = [
    "Analyze the following situation: {scenario}. What should be done?",
    "What are the potential consequences of {action}?",
    "Compare and contrast {approach1} vs {approach2} for solving {problem}.",
    "Given {facts}, what can we conclude about {question}?",
    "Identify the logical fallacy in: {statement}",
]

# -----------------------------------------------------------------------------
# Example generators


def _render_prompt(template: str, **values: str) -> str:
    """Fill a prompt template and reject unresolved fields."""
    try:
        prompt = template.format(**values)
    except KeyError as exc:
        raise RuntimeError(
            f"Prompt template is missing a value for {exc.args[0]!r}"
        ) from exc
    if "{" in prompt or "}" in prompt:
        raise RuntimeError(f"Prompt template was not fully rendered: {prompt!r}")
    return prompt


def generate_general_chat_prompt() -> str:
    """Generate a general chat/instruction prompt."""
    template = random.choice(GENERAL_CHAT_PROMPTS)
    topics = [
        "project updates",
        "meeting schedules",
        "team collaboration",
        "personal development",
    ]
    concepts = ["machine learning", "photosynthesis", "quantum computing", "blockchain"]
    items = [("Python", "JavaScript"), ("SQL", "NoSQL"), ("REST", "GraphQL")]
    activities = [
        "learning a new language",
        "time management",
        "public speaking",
        "remote work",
    ]
    tasks = ["bake bread", "start a podcast", "learn photography", "meditate daily"]
    item1, item2 = random.choice(items)
    return _render_prompt(
        template,
        topic=random.choice(topics),
        concept=random.choice(concepts),
        item1=item1,
        item2=item2,
        activity=random.choice(activities),
        task=random.choice(tasks),
    )


def generate_code_prompt() -> str:
    """Generate a code-related prompt."""
    template = random.choice(CODE_PROMPTS)
    tasks = [
        "calculates factorial recursively",
        "sorts a list using quicksort",
        "finds prime numbers",
    ]
    algorithms = [
        "binary search",
        "merge sort",
        "breadth-first search",
        "dynamic programming",
    ]
    languages = ["Python", "JavaScript", "Java", "C++"]
    buggy_code = [
        "def add(a, b):\n    return a - b",
        "for i in range(3)\n    print(i)",
    ]
    code = [
        "def unique(values):\n    return list(set(values))",
        "total = sum(value for value in values if value > 0)",
    ]
    return _render_prompt(
        template,
        task=random.choice(tasks),
        buggy_code=random.choice(buggy_code),
        algorithm=random.choice(algorithms),
        language=random.choice(languages),
        functionality=random.choice(tasks),
        code=random.choice(code),
    )


def generate_math_prompt() -> str:
    """Generate a math word problem."""
    template = random.choice(MATH_PROMPTS)

    problems = [
        "Tom has 5 apples and buys 3 more. How many apples does he have?",
        "A train travels at 60 mph for 2.5 hours. How far does it travel?",
        "Calculate 15% of 240.",
    ]
    conditions = [
        ("a book costs $24 after a 20% discount", "its original price"),
        ("three notebooks cost $12", "the cost of five notebooks"),
    ]
    scenarios = [
        ("basket", "12 oranges", "4 are removed", "the number of oranges left"),
        ("tank", "30 liters of water", "8 liters are added", "the new volume"),
    ]
    equations = ["3x + 5 = 20", "2x - 7 = 11"]
    value_pairs = [(80, 100), (120, 150)]
    condition, query = random.choice(conditions)
    scenario, initial_state, change, final_query = random.choice(scenarios)
    value1, value2 = random.choice(value_pairs)
    return _render_prompt(
        template,
        problem=random.choice(problems),
        condition=condition,
        query=query,
        scenario=scenario,
        initial_state=initial_state,
        change=change,
        final_query=final_query,
        equation=random.choice(equations),
        value1=str(value1),
        value2=str(value2),
    )


def generate_reasoning_prompt() -> str:
    """Generate a reasoning/analysis prompt."""
    template = random.choice(REASONING_PROMPTS)
    scenarios = [
        "A company is losing market share to competitors",
        "Climate change is affecting crop yields",
        "AI is being adopted in healthcare",
    ]
    approach_pairs = [("batch processing", "stream processing"), ("buying", "building")]
    facts_and_questions = [
        ("all robins are birds and this animal is a robin", "the animal's class"),
        (
            "the server is healthy only when both probes pass, and one probe failed",
            "server health",
        ),
    ]
    statements = [
        "Everyone I asked likes the product, so everyone likes it.",
        "We must choose this plan because the alternative is total failure.",
    ]
    approach1, approach2 = random.choice(approach_pairs)
    facts, question = random.choice(facts_and_questions)
    return _render_prompt(
        template,
        scenario=random.choice(scenarios),
        action="automating a manual review process",
        approach1=approach1,
        approach2=approach2,
        problem="processing a growing event stream",
        facts=facts,
        question=question,
        statement=random.choice(statements),
    )


# -----------------------------------------------------------------------------
# Teacher model interaction


def query_teacher_model(
    prompt: str,
    use_cot: bool = False,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    client=None,
) -> str:
    """
    Query the teacher model (e.g., OpenAI GPT) to generate a response.

    Args:
        prompt: The user prompt to send to the teacher
        use_cot: Whether to request chain-of-thought reasoning
        model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        The generated response from the teacher model
    """
    if client is None:
        client = create_openai_client()

    try:
        # Add CoT instruction if requested
        system_message = "You are a helpful, accurate, and concise assistant."
        if use_cot:
            system_message += " Think step-by-step and show your reasoning before providing the final answer."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Teacher model returned an empty response")
        return content

    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError("Teacher model request failed") from e


# -----------------------------------------------------------------------------
# Data generation pipeline


def generate_example(
    task_type: str,
    use_cot: bool = False,
    teacher_model: str = "gpt-4o-mini",
    client=None,
) -> Dict:
    """
    Generate a single training example.

    Returns a dict with format compatible with tasks/customjson.py:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    # Generate prompt based on task type
    if task_type == "general_chat":
        user_prompt = generate_general_chat_prompt()
    elif task_type == "code":
        user_prompt = generate_code_prompt()
    elif task_type == "math":
        user_prompt = generate_math_prompt()
    elif task_type == "reasoning":
        user_prompt = generate_reasoning_prompt()
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Get response from teacher model
    assistant_response = query_teacher_model(
        user_prompt,
        use_cot=use_cot,
        model=teacher_model,
        client=client,
    )

    # Format as conversation
    example = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful, accurate, and friendly assistant.",
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }

    return example


def generate_dataset(
    num_examples: int,
    output_file: str,
    task_mixture: Dict[str, float] = TASK_MIXTURE,
    use_cot_ratio: float = 0.3,
    teacher_model: str = "gpt-4o-mini",
    resume: bool = True,
    client=None,
):
    """
    Generate a full dataset of teacher-distilled examples.

    Args:
        num_examples: Number of examples to generate
        output_file: Path to output JSONL file
        task_mixture: Dict mapping task types to their proportions
        use_cot_ratio: Proportion of examples that should use chain-of-thought
        teacher_model: Teacher model identifier
        resume: Whether to resume from existing file
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

            # Decide whether to use CoT
            use_cot = random.random() < use_cot_ratio

            example = generate_example(
                task_type,
                use_cot=use_cot,
                teacher_model=teacher_model,
                client=client,
            )

            # Only complete provider responses are persisted.
            f.write(json.dumps(example) + "\n")
            f.flush()  # Ensure it's written in case of interruption

            if (i + 1) % 10 == 0:
                print(
                    f"Generated {i + 1}/{num_examples} examples (task: {task_type}, cot: {use_cot})"
                )

    print(f"\n✅ Dataset generation complete! Saved to {output_file}")
    print(f"   Total examples: {num_examples}")
    print(f"   Task mixture: {task_mixture}")
    print(f"   CoT ratio: {use_cot_ratio}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher-distilled SFT data")
    parser.add_argument(
        "--output-file",
        type=str,
        default="teacher_sft.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--num-examples", type=int, default=1000, help="Number of examples to generate"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gpt-4o-mini",
        help="Teacher model to use (e.g., gpt-4, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--use-cot-ratio",
        type=float,
        default=0.3,
        help="Proportion of examples with chain-of-thought (0.0-1.0)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from existing file"
    )

    args = parser.parse_args()

    client = create_openai_client()

    # Ensure output directory exists
    base_dir = get_base_dir()
    output_file = os.path.join(base_dir, args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generate dataset
    generate_dataset(
        num_examples=args.num_examples,
        output_file=output_file,
        use_cot_ratio=args.use_cot_ratio,
        teacher_model=args.teacher_model,
        resume=not args.no_resume,
        client=client,
    )
