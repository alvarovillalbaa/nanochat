import math

import pytest
import torch

from nanochat.dpo import calculate_num_iterations, dpo_loss, sequence_log_probs


def test_calculate_num_iterations_includes_small_and_final_batches():
    assert calculate_num_iterations(1, target_examples_per_step=16, num_epochs=1) == 1
    assert calculate_num_iterations(32, target_examples_per_step=16, num_epochs=2) == 4
    assert (
        calculate_num_iterations(
            32,
            target_examples_per_step=16,
            num_epochs=2,
            max_iterations=3,
        )
        == 3
    )


@pytest.mark.parametrize(
    "num_examples,target_examples,num_epochs", [(0, 16, 1), (1, 0, 1), (1, 16, 0)]
)
def test_calculate_num_iterations_rejects_empty_training_runs(
    num_examples,
    target_examples,
    num_epochs,
):
    with pytest.raises(ValueError):
        calculate_num_iterations(num_examples, target_examples, num_epochs)


def test_sequence_log_probs_masks_ignored_targets_per_sequence():
    class FakeModel:
        def __call__(self, inputs, targets, loss_reduction):
            assert loss_reduction == "none"
            return torch.tensor([1.0, 99.0, 2.0, 3.0])

    inputs = torch.ones((2, 2), dtype=torch.long)
    targets = torch.tensor([[4, -1], [5, 6]])

    result = sequence_log_probs(FakeModel(), inputs, targets)

    assert result.tolist() == [-1.0, -5.0]


def test_dpo_loss_uses_frozen_reference_log_ratios():
    policy_chosen = torch.tensor([0.0], requires_grad=True)
    policy_rejected = torch.tensor([0.0], requires_grad=True)
    reference_chosen = torch.tensor([0.0], requires_grad=True)
    reference_rejected = torch.tensor([0.0], requires_grad=True)

    loss, metrics = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=1.0,
    )
    loss.backward()

    assert loss.item() == pytest.approx(math.log(2))
    assert metrics["reward_margin"].item() == pytest.approx(0.0)
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None


def test_dpo_loss_rewards_policy_improvement_over_reference():
    loss, metrics = dpo_loss(
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )

    assert loss.item() < math.log(2)
    assert metrics["chosen_reward"].item() == pytest.approx(2.0)
    assert metrics["rejected_reward"].item() == pytest.approx(0.0)
    assert metrics["reward_margin"].item() == pytest.approx(2.0)
    assert metrics["preference_accuracy"].item() == pytest.approx(1.0)
