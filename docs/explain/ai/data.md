# Data Pipeline

The data pipeline is designed to efficiently stream large-scale pretraining datasets. Implementation is found in `nanochat/dataset.py`.

## Overview

```mermaid
graph LR
    HF[HuggingFace Hub] -->|Download| Local[Local Disk (base_data/)]
    Local -->|Read| Parquet[Parquet Files]
    Parquet -->|Stream| Iterator[Batched Iterator]
    Iterator -->|Tokenize| Loader[DataLoader]
    Loader -->|Batch| GPU[GPU Training]
```

## Dataset Source

*   **Primary Dataset**: [FineWeb-Edu 100B](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) (shuffled).
*   **Format**: Parquet files.
*   **Hosting**: HuggingFace Datasets.

## Streaming & Caching

The system does not require downloading the entire dataset upfront. Dataset
preparation and training are separate steps:

1.  **Explicit Download**: `python -m nanochat.dataset -n N` downloads the requested shards.
2.  **Local Cache**: Downloaded shards are stored in `base_data/` (relative to the project root).
3.  **Batched Iteration**: Training reads the already-local files row-group by row-group. It does not fetch a missing shard from inside the iterator.

## Structure

*   **Shards**: The dataset is split into ~1800 shards (`shard_00000.parquet` to `shard_01822.parquet`).
*   **Split**:
    *   **Train**: All shards except the last one.
    *   **Validation**: The last shard.

## Usage

To manually download shards (e.g., to prep a machine):
```bash
python -m nanochat.dataset -n 10 -w 4
```
This downloads the first 10 shards using 4 parallel workers.
