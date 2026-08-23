# Optimization Strategy

The training process in `nanochat` employs a sophisticated, hybrid optimization strategy to maximize convergence speed and stability. This is implemented in `nanochat/gpt.py` (setup) and `nanochat/muon.py` (optimizer).

## Hybrid Optimization

The model parameters are split into two groups, each optimized by a different algorithm:

```mermaid
graph TD
    Params[All Model Parameters] --> Split{Parameter Type?}

    Split -->|2D Matrices (Linear, Attn)| MuonGroup[Muon Optimizer]
    Split -->|Embeddings, Norms, Biases| AdamGroup[AdamW Optimizer]

    MuonGroup -->|SGD + Momentum| Update1[Update Step]
    Update1 -->|Newton-Schulz| Ortho[Orthogonalization]

    AdamGroup -->|Standard AdamW| Update2[Update Step]
```

### 1. Muon (MomentUm Orthogonalized by Newton-schulz)
**Target**: 2D parameters (Linear layers, attention projections).

**Muon** is a novel optimizer that combines SGD with momentum and an orthogonalization step.
*   **Mechanism**: After a standard SGD-momentum update, the update matrix is orthogonalized using **Newton-Schulz iteration**.
*   **Why?**: This ensures that the updates are spectrally normalized, which can lead to faster convergence for the bulk of the transformer's weights.
*   **Implementation**: The Newton-Schulz iteration is implemented efficiently in `bfloat16` on the GPU.
*   **Math**: The update $G$ is orthogonalized via the iteration:
    $$ X_0 = G $$
    $$ X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k) $$
    *(Note: The actual implementation uses a quintic polynomial for faster convergence)*

*   **Distributed**: The `DistMuon` implementation handles distributed training (DDP) by performing `reduce_scatter` (to average gradients) and `all_gather` (to sync weights) manually.

### 2. AdamW
**Target**: Embeddings (`wte`), Language Model Head (`lm_head`), and any scalars/biases.

Standard **AdamW** is used for parameters that are not suitable for orthogonalization (like embeddings which are effectively 0D/1D vectors per token, or the final classifier).
*   **Learning Rate Scaling**: The learning rate for AdamW groups is scaled by $1/\sqrt{d_{model}}$ to align with the scaling laws of the architecture.

## Hyperparameters

*   **Muon LR**: Typically 0.02 (higher than standard AdamW rates).
*   **Muon Momentum**: 0.95.
*   **AdamW Betas**: (0.8, 0.95).
*   **Weight Decay**: Applied via AdamW (default 0.0 in the setup, but configurable).

## Distributed Training

The optimization setup is fully compatible with Distributed Data Parallel (DDP).
*   **Muon**: Has a specialized `DistMuon` class that optimizes communication by overlapping gradient reduction with computation where possible.
*   **AdamW**: Uses `DistAdamW` (if available) or standard PyTorch `AdamW` with fused kernels.
