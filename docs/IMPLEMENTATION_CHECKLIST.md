# Training-extension status

This checklist distinguishes implemented code from unverified research ideas.

## Implemented and focused-tested

- [x] Optional teacher-data generator with explicit SDK/key/provider failures.
- [x] Validated `CustomJSON` loader for SFT conversations.
- [x] Teacher data is opt-in in `chat_sft.py`.
- [x] Optional weighted `TaskMixture` sampling with empty-task validation.
- [x] Preference-pair generation with validated provider rankings.
- [x] DPO from the canonical `sft` checkpoint source.
- [x] Frozen reference model and reference-relative DPO objective.
- [x] Supervised-token sequence log-probability aggregation.
- [x] Final DPO optimizer step is executed and saved.
- [x] Opt-in SwiGLU architecture recorded in checkpoint metadata.
- [x] Legacy checkpoint metadata defaults to the original ReLU-squared shape.

## Implemented but not end-to-end verified

- [ ] Review a representative generated SFT dataset for quality and safety.
- [ ] Review a representative generated preference dataset for ranking quality.
- [ ] Run a small GPU SFT smoke training job.
- [ ] Run a small GPU DPO smoke training job and reload its checkpoint.
- [ ] Compare matched baseline and experimental evaluation metrics.
- [ ] Measure GPU memory and throughput.

## Not implemented

- [ ] Explicit tool-call schema and execution-trace dataset.
- [ ] Long-context training and evaluation.
- [ ] Sparse mixture-of-experts layers.
- [ ] Scaling-law fitting or automatic early stopping.
- [ ] Dedicated held-out preference evaluation.

RMSNorm, rotary embeddings, multi-query attention, bias-free linear layers, and
scaled dot-product attention already existed in the upstream implementation;
they are not changes introduced by this branch. See
[`IMPROVEMENTS.md`](IMPROVEMENTS.md) for commands and compatibility details.
