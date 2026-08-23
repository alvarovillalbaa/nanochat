# External Libraries & Dependencies

This document lists the key external libraries used in `nanochat` and their role in the system.

## Core ML & Math

*   **[PyTorch](https://pytorch.org/) (`torch`)**: The primary deep learning framework. Used for tensor operations, autograd, and model definition.
    *   *Version*: Targeted at CUDA 12.8 compatible versions.
*   **[NumPy](https://numpy.org/) (`numpy`)**: Fundamental package for scientific computing. Used for data manipulation and interfacing with other libraries.

## Tokenization & NLP

*   **[Tiktoken](https://github.com/openai/tiktoken) (`tiktoken`)**: A fast BPE tokeniser for use with OpenAI's models. Used here for high-performance inference tokenization.
*   **[Tokenizers](https://github.com/huggingface/tokenizers) (`tokenizers`)**: HuggingFace's fast tokenizer library (Rust bindings). Used for training the tokenizer and compatibility.
*   **[Regex](https://pypi.org/project/regex/) (`regex`)**: Advanced regular expressions, required for the GPT-4 splitting pattern.

## Data & System

*   **[Datasets](https://huggingface.co/docs/datasets) (`datasets`)**: HuggingFace library used by task datasets such as ARC, MMLU, GSM8K, HumanEval, and SmolTalk.
*   **[PyArrow](https://arrow.apache.org/docs/python/) (`pyarrow`)**: (Implicit dependency of `datasets`/parquet). Used for reading Parquet files efficiently.
*   **[PSUtil](https://psutil.readthedocs.io/) (`psutil`)**: Process and system utilities used by report generation.

## Web & API

*   **[FastAPI](https://fastapi.tiangolo.com/) (`fastapi`)**: Modern, fast web framework for building APIs with Python. Used for the inference server/UI backend.
*   **[Uvicorn](https://www.uvicorn.org/) (`uvicorn`)**: ASGI web server implementation. Runs the FastAPI application.

## Tracking & Utilities

*   **[Weights & Biases](https://wandb.ai/) (`wandb`)**: Experiment tracking and visualization. Used to log training metrics (loss, throughput, etc.).
*   **[Files-to-Prompt](https://github.com/simonw/files-to-prompt) (`files-to-prompt`)**: Utility used by the README's optional source-packaging workflow.

## Internal/Custom

*   **`rustbpe`**: A custom Rust extension (built with `maturin`) for tokenizer training.
