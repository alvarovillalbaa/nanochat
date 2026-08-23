from nanochat import checkpoint_manager
from nanochat.gpt import GPT, GPTConfig, MLP


def test_legacy_checkpoint_config_defaults_to_relu_squared():
    legacy_config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=8,
    )

    mlp = MLP(legacy_config)

    assert legacy_config.use_swiglu is False
    assert hasattr(mlp, "c_fc")
    assert not hasattr(mlp, "c_fc1")


def test_swiglu_remains_available_when_explicitly_enabled():
    config = GPTConfig(n_embd=8, use_swiglu=True)

    mlp = MLP(config)

    assert hasattr(mlp, "c_fc1")
    assert hasattr(mlp, "c_fc2")
    assert not hasattr(mlp, "c_fc")


def test_legacy_checkpoint_state_dict_loads_strictly_without_swiglu_metadata():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=8,
    )
    original = GPT(config)
    legacy_metadata = config.__dict__.copy()
    legacy_metadata.pop("use_swiglu")

    restored = GPT(GPTConfig(**legacy_metadata))
    restored.load_state_dict(original.state_dict(), strict=True)


def test_dpo_checkpoint_source_can_be_reloaded(monkeypatch, tmp_path):
    captured = {}

    def fake_load_model_from_dir(checkpoints_dir, *args, **kwargs):
        captured["checkpoints_dir"] = checkpoints_dir
        return object(), object(), {}

    monkeypatch.setattr(checkpoint_manager, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        checkpoint_manager,
        "load_model_from_dir",
        fake_load_model_from_dir,
    )

    checkpoint_manager.load_model("dpo", "cpu", phase="eval")

    assert captured["checkpoints_dir"] == str(tmp_path / "chatdpo_checkpoints")
