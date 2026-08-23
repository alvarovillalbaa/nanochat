"""Pure helpers for Direct Preference Optimization training."""

import math

import torch
import torch.nn.functional as F


def calculate_num_iterations(
    num_examples: int,
    target_examples_per_step: int,
    num_epochs: int,
    max_iterations: int = -1,
) -> int:
    """Return the number of optimizer steps, including a partial final batch."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if target_examples_per_step <= 0:
        raise ValueError("target_examples_per_step must be positive")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if max_iterations == 0 or max_iterations < -1:
        raise ValueError("max_iterations must be -1 or a positive integer")

    num_iterations = math.ceil(num_examples / target_examples_per_step) * num_epochs
    if max_iterations > 0:
        num_iterations = min(num_iterations, max_iterations)
    return num_iterations


def sequence_log_probs(
    model, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return the summed supervised-token log probability for each sequence."""
    token_nll = model(inputs, targets, loss_reduction="none").view_as(targets)
    supervised = targets.ne(-1)
    return -(token_nll * supervised).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the reference-relative DPO loss and detached logging metrics."""
    if beta <= 0:
        raise ValueError("beta must be positive")

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps.detach())
    rejected_rewards = beta * (
        policy_rejected_logps - reference_rejected_logps.detach()
    )
    reward_margins = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(reward_margins).mean()

    metrics = {
        "chosen_reward": chosen_rewards.detach().mean(),
        "rejected_reward": rejected_rewards.detach().mean(),
        "reward_margin": reward_margins.detach().mean(),
        "preference_accuracy": (chosen_rewards > rejected_rewards)
        .float()
        .detach()
        .mean(),
    }
    return loss, metrics
