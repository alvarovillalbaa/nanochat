"""Train a chat checkpoint with Direct Preference Optimization (DPO).

Run on one GPU for debugging:

    python -m scripts.chat_dpo

Or use ``torchrun`` for distributed training. The input JSONL must contain
``prompt``, ``chosen``, and ``rejected`` strings for every row.
"""

import copy
import json
import os

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.dpo import calculate_num_iterations, dpo_loss, sequence_log_probs
from nanochat.engine import Engine
from nanochat.report import get_report
from scripts.chat_eval import run_chat_eval

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# DPO hyperparameters
run = "dummy"  # wandb run name ("dummy" disables wandb logging)
# Input model options
source = "sft"  # base|mid|sft
model_tag = None
step = None
# Compute/precision
dtype = "bfloat16"
device_batch_size = 2  # preference pairs require extra memory
# Optimization
num_epochs = 1
max_iterations = -1
target_examples_per_step = 16
unembedding_lr = 0.002
embedding_lr = 0.1
matrix_lr = 0.01
weight_decay = 0.0
init_lr_frac = 0.02
beta = 0.1
# Evaluation
eval_metrics_every = 200
# Data
preference_data_file = "preference_data.jsonl"

# Allow CLI overrides through nanochat's configurator.
config_keys = [
    key
    for key, value in globals().items()
    if not key.startswith("_") and isinstance(value, (int, float, bool, str))
]
exec(open(os.path.join("nanochat", "configurator.py")).read())
user_config = {key: globals()[key] for key in config_keys}
checkpoint_step = step


def load_preference_data(filepath: str) -> list[dict[str, str]]:
    """Load and validate preference pairs from a JSONL file."""
    full_path = (
        filepath if os.path.isabs(filepath) else os.path.join(get_base_dir(), filepath)
    )
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Preference dataset not found: {full_path}. Generate it with "
            f"python -m scripts.gen_preference_data --output-file {filepath}"
        )

    data = []
    with open(full_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in preference dataset at line {line_number}"
                ) from exc

            if not isinstance(example, dict):
                raise ValueError(
                    f"Preference dataset line {line_number} must contain a JSON object"
                )

            if {"prompt", "chosen", "rejected"} <= example.keys():
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]
            elif {"messages", "chosen", "rejected"} <= example.keys():
                messages = example["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"Preference dataset line {line_number} has no prompt messages"
                    )
                if not isinstance(messages[-1], dict):
                    raise ValueError(
                        f"Preference dataset line {line_number} has an invalid prompt message"
                    )
                prompt = messages[-1].get("content")
                chosen_value = example["chosen"]
                rejected_value = example["rejected"]
                chosen = (
                    chosen_value.get("content")
                    if isinstance(chosen_value, dict)
                    else chosen_value
                )
                rejected = (
                    rejected_value.get("content")
                    if isinstance(rejected_value, dict)
                    else rejected_value
                )
            else:
                raise ValueError(
                    f"Preference dataset line {line_number} is missing required fields"
                )

            values = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            if not all(
                isinstance(value, str) and value.strip() for value in values.values()
            ):
                raise ValueError(
                    f"Preference dataset line {line_number} contains an empty field"
                )
            data.append(values)

    if not data:
        raise ValueError(f"Preference dataset is empty: {full_path}")
    print0(f"Loaded {len(data)} preference pairs from {filepath}")
    return data


preference_data = load_preference_data(preference_data_file)


# Compute initialization
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# Logging and model initialization
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(
        project="nanochat-dpo",
        name=run,
        config=user_config,
        save_code=True,
    )
)
model, tokenizer, meta = load_model(
    source,
    device,
    phase="train",
    model_tag=model_tag,
    step=checkpoint_step,
)
reference_model = copy.deepcopy(model).eval()
reference_model.requires_grad_(False)
engine = Engine(model, tokenizer)


def dpo_data_generator(dataset, batch_size):
    """Yield tokenized chosen and rejected batches indefinitely."""
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_responses(response_batch):
        nrows = len(response_batch)
        ncols = max(len(ids) for ids, _ in response_batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)

        for row, (ids, mask) in enumerate(response_batch):
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[row, : len(ids) - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            row_mask = torch.tensor(mask[1:], dtype=torch.bool)
            row_targets[~row_mask] = -1
            targets[row, : len(ids) - 1] = row_targets

        return inputs.to(device), targets.to(device)

    batch = []
    while True:
        for index in range(ddp_rank, len(dataset), ddp_world_size):
            example = dataset[index]
            prefix = [
                {
                    "role": "system",
                    "content": "You are a helpful, accurate, and friendly assistant.",
                },
                {"role": "user", "content": example["prompt"]},
            ]
            chosen = [*prefix, {"role": "assistant", "content": example["chosen"]}]
            rejected = [
                *prefix,
                {"role": "assistant", "content": example["rejected"]},
            ]
            chosen_ids, chosen_mask = tokenizer.render_conversation(chosen)
            rejected_ids, rejected_mask = tokenizer.render_conversation(rejected)
            batch.append(
                (
                    (chosen_ids, chosen_mask),
                    (rejected_ids, rejected_mask),
                )
            )
            if len(batch) == batch_size:
                chosen_batch = [chosen_row for chosen_row, _ in batch]
                rejected_batch = [rejected_row for _, rejected_row in batch]
                yield collate_responses(chosen_batch), collate_responses(rejected_batch)
                batch = []


examples_per_micro_step = device_batch_size * ddp_world_size
if target_examples_per_step % examples_per_micro_step != 0:
    raise ValueError(
        "target_examples_per_step must be divisible by device_batch_size * world_size"
    )
if len(preference_data) < ddp_world_size:
    raise ValueError("Preference data must include at least one example per worker")

grad_accum_steps = target_examples_per_step // examples_per_micro_step
num_iterations = calculate_num_iterations(
    len(preference_data),
    target_examples_per_step,
    num_epochs,
    max_iterations,
)
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per micro-step: {examples_per_micro_step}")
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Optimizer iterations: {num_iterations}")
train_iter = iter(dpo_data_generator(preference_data, batch_size=device_batch_size))

# Optimizer setup
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
for optimizer in optimizers:
    for group in optimizer.param_groups:
        group["lr"] *= init_lr_frac
        group["initial_lr"] = group["lr"]


def get_lr_multiplier(iteration: int) -> float:
    """Linearly decay the learning rate while keeping the final step positive."""
    return 1.0 - iteration / num_iterations


def evaluate_chat_metrics(iteration: int) -> dict[str, float]:
    """Run the existing compact chat benchmarks against the policy model."""
    model.eval()
    with torch.no_grad(), autocast_ctx:
        metrics = {
            "mmlu_acc": run_chat_eval(
                "MMLU",
                model,
                tokenizer,
                engine,
                batch_size=device_batch_size * 2,
                max_problems=512,
            ),
            "arc_easy_acc": run_chat_eval(
                "ARC-Easy",
                model,
                tokenizer,
                engine,
                batch_size=device_batch_size * 2,
                max_problems=512,
            ),
        }
    print0(
        f"Step {iteration:05d} | "
        + ", ".join(f"{key}: {value:.6f}" for key, value in metrics.items())
    )
    wandb_run.log({"step": iteration, **metrics})
    model.train()
    return metrics


final_metrics = {}
final_training_metrics = {}
for train_step in range(num_iterations):
    if train_step > 0 and train_step % eval_metrics_every == 0:
        final_metrics = evaluate_chat_metrics(train_step)

    total_loss = torch.zeros((), device=device)
    metric_totals = {
        "chosen_reward": torch.zeros((), device=device),
        "rejected_reward": torch.zeros((), device=device),
        "reward_margin": torch.zeros((), device=device),
        "preference_accuracy": torch.zeros((), device=device),
    }

    for _ in range(grad_accum_steps):
        (chosen_inputs, chosen_targets), (rejected_inputs, rejected_targets) = next(
            train_iter
        )
        with autocast_ctx:
            policy_chosen_logps = sequence_log_probs(
                model, chosen_inputs, chosen_targets
            )
            policy_rejected_logps = sequence_log_probs(
                model, rejected_inputs, rejected_targets
            )
        with torch.no_grad(), autocast_ctx:
            reference_chosen_logps = sequence_log_probs(
                reference_model, chosen_inputs, chosen_targets
            )
            reference_rejected_logps = sequence_log_probs(
                reference_model, rejected_inputs, rejected_targets
            )

        loss, batch_metrics = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=beta,
        )
        normalized_loss = loss / grad_accum_steps
        normalized_loss.backward()
        total_loss += normalized_loss.detach()
        for key, value in batch_metrics.items():
            metric_totals[key] += value / grad_accum_steps

    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        for value in metric_totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.AVG)

    lr_multiplier = get_lr_multiplier(train_step)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
        optimizer.step()
    model.zero_grad(set_to_none=True)

    final_training_metrics = {
        "train_loss": total_loss.item(),
        **{key: value.item() for key, value in metric_totals.items()},
    }
    print0(
        f"Step {train_step + 1:05d}/{num_iterations:05d} | "
        f"Loss: {final_training_metrics['train_loss']:.6f} | "
        f"Reward margin: {final_training_metrics['reward_margin']:.6f} | "
        f"Preference accuracy: {final_training_metrics['preference_accuracy']:.6f} | "
        f"lrm: {lr_multiplier:.6f}"
    )
    wandb_run.log(
        {
            "step": train_step + 1,
            **final_training_metrics,
            "lrm": lr_multiplier,
        }
    )

completed_steps = num_iterations
final_metrics = evaluate_chat_metrics(completed_steps)

if master_process:
    depth = model.config.n_layer
    output_model_tag = f"d{depth}"
    checkpoint_dir = os.path.join(
        get_base_dir(), "chatdpo_checkpoints", output_model_tag
    )
    save_checkpoint(
        checkpoint_dir,
        completed_steps,
        model.state_dict(),
        None,
        {
            "step": completed_steps,
            **final_training_metrics,
            **final_metrics,
            "model_config": model.config.__dict__,
        },
    )
    print(f"Saved DPO model checkpoint to {checkpoint_dir}")

get_report().log(
    section="Chat DPO",
    data=[
        user_config,
        {
            "Preference pairs": len(preference_data),
            "Iterations": completed_steps,
            "Final loss": final_training_metrics["train_loss"],
            "Final reward margin": final_training_metrics["reward_margin"],
        },
    ],
)

wandb_run.finish()
compute_cleanup()
