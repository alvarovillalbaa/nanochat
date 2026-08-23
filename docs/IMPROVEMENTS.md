# Optional Training Extensions

This branch adds three experimental, opt-in capabilities while preserving the
original nanochat defaults:

- teacher-generated JSONL data for SFT;
- offline preference-pair generation and reference-relative DPO training;
- a SwiGLU MLP architecture option for newly trained checkpoints.

No quality or speed gain is claimed without a matched training and evaluation
run. The focused unit tests cover configuration compatibility, data validation,
and DPO loss semantics; they do not substitute for GPU training evidence.

## Teacher-generated SFT data

The provider client is optional, so install it separately before generation:

```bash
pip install openai
export OPENAI_API_KEY='your-key-here'
python -m scripts.gen_teacher_data \
    --output-file teacher_sft.jsonl \
    --num-examples 10000 \
    --teacher-model gpt-4o-mini \
    --use-cot-ratio 0.3
```

Generation fails before writing a row when the SDK or API key is missing, the
provider request fails, or the provider returns an empty response. Resume mode
only counts rows already present; review generated data before training.

Teacher data is disabled by default. Opt in explicitly:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft \
    --use_teacher_data=True \
    --teacher_data_file=teacher_sft.jsonl \
    --teacher_data_weight=1.0
```

`CustomJSON` rejects a missing, empty, malformed, or structurally invalid
dataset. A positive-weight empty task is also rejected before sampling.

## Preference data and DPO

Generate validated chosen/rejected pairs with the same optional provider:

```bash
python -m scripts.gen_preference_data \
    --output-file preference_data.jsonl \
    --num-examples 1000 \
    --num-candidates 3
```

Candidate generation and ranking fail closed. In particular, an incomplete or
invalid ranking is not replaced with a random label.

Run DPO from an SFT checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_dpo \
    --source=sft \
    --preference_data_file=preference_data.jsonl \
    --beta=0.1
```

The policy is optimized against a frozen copy of the starting SFT model using
the standard reference-relative log-ratio objective. The script sums log
probabilities only over supervised assistant tokens, includes a partial final
training batch in its step count, and executes the final optimizer step before
saving.

DPO holds both a policy and frozen reference model and evaluates chosen and
rejected sequences, so memory use is higher than SFT. Reduce
`device_batch_size` if needed. End-to-end GPU training has not been run in this
branch.

Saved DPO checkpoints can be selected as `dpo` in the chat CLI, web app, and
evaluation script.

## SwiGLU checkpoint compatibility

`GPTConfig.use_swiglu` defaults to `False`. This preserves the parameter names
and shapes expected by existing metadata that predates the field. To train a
new SwiGLU model, opt in during base training:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --use_swiglu=True
```

The architecture choice is written to base and mid-training checkpoint
metadata. All later phases must continue with the architecture stored in the
checkpoint; changing the flag while resuming is not a checkpoint conversion.

## Weighted task mixtures

`TaskMixture` accepts optional non-negative weights:

```python
train_ds = TaskMixture([arc, gsm8k, smoltalk], weights=[0.4, 0.3, 0.3])
```

Weights are normalized. Their count must match the task count, their sum must
be positive, and a task with positive weight must contain at least one example.

## Verification boundary

The focused local checks are documented in the change handoff. Still required
before treating these extensions as production-ready:

- a representative teacher/preference data quality review;
- a small GPU SFT and DPO smoke run with real checkpoints;
- matched baseline-versus-extension evaluation;
- memory and throughput measurements for the intended hardware.

The broader ideas in [`roadmap/init_analysis.md`](roadmap/init_analysis.md) are
research directions, not implemented or measured capabilities.
