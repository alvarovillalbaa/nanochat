"""
Custom JSONL dataset loader for teacher-distilled or custom SFT data.
Expects JSONL format where each line is a conversation with "messages" field.

Example format:
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
}
"""

import json
import os
from nanochat.common import get_base_dir


class CustomJSON:
    """Load custom JSONL conversation data."""

    def __init__(self, filepath: str = None, split: str = "train", stop: int = None):
        """
        Args:
            filepath: Path to JSONL file (relative to base_dir or absolute)
            split: "train" or "test" (for compatibility, currently returns same data)
            stop: Optional limit on number of examples to load
        """
        if filepath is None:
            filepath = "teacher_sft.jsonl"

        # Handle relative vs absolute paths
        if not os.path.isabs(filepath):
            base_dir = get_base_dir()
            filepath = os.path.join(base_dir, filepath)

        self.filepath = filepath
        self.split = split
        self.stop = stop
        self.data = []

        # Load the data
        self._load_data()

    def _load_data(self):
        """Load conversations from JSONL file."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Custom JSONL dataset not found: {self.filepath}. Generate it with "
                f"python -m scripts.gen_teacher_data --output-file {os.path.basename(self.filepath)}"
            )

        with open(self.filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.stop is not None and i >= self.stop:
                    break

                try:
                    example = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {i + 1} of {self.filepath}"
                    ) from e

                if not isinstance(example, dict):
                    raise ValueError(
                        f"Line {i + 1} of {self.filepath} must contain a JSON object"
                    )
                messages = example.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"Line {i + 1} of {self.filepath} must contain a non-empty messages list"
                    )
                if not all(
                    isinstance(message, dict)
                    and isinstance(message.get("role"), str)
                    and isinstance(message.get("content"), str)
                    and message["content"].strip()
                    for message in messages
                ):
                    raise ValueError(
                        f"Line {i + 1} of {self.filepath} contains an invalid message"
                    )
                self.data.append(messages)

        if not self.data:
            raise ValueError(f"Custom JSONL dataset is empty: {self.filepath}")

        print(f"Loaded {len(self.data)} examples from {self.filepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a conversation in the format expected by the tokenizer.
        Returns a list of message dicts with 'role' and 'content' keys.
        """
        return self.data[idx]
