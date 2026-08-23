from types import SimpleNamespace

import pytest

from scripts import gen_preference_data, gen_teacher_data
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON


class _Task:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def test_weighted_task_mixture_rejects_empty_positive_weight_task():
    with pytest.raises(ValueError, match="positive-weight task.*empty"):
        TaskMixture([_Task(0), _Task(4)], weights=[0.5, 0.5])


def test_weighted_task_mixture_keeps_small_positive_weight_task():
    mixture = TaskMixture([_Task(1), _Task(10)], weights=[0.1, 0.9])

    assert {task_index for task_index, _ in mixture.index_map} == {0, 1}


def test_custom_json_missing_file_fails_closed(tmp_path):
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Custom JSONL dataset not found"):
        CustomJSON(filepath=str(missing_path))


def test_custom_json_empty_file_fails_closed(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("")

    with pytest.raises(ValueError, match="dataset is empty"):
        CustomJSON(filepath=str(empty_path))


@pytest.mark.parametrize(
    "templates,generator",
    [
        (
            gen_teacher_data.GENERAL_CHAT_PROMPTS,
            gen_teacher_data.generate_general_chat_prompt,
        ),
        (gen_teacher_data.CODE_PROMPTS, gen_teacher_data.generate_code_prompt),
        (gen_teacher_data.MATH_PROMPTS, gen_teacher_data.generate_math_prompt),
        (
            gen_teacher_data.REASONING_PROMPTS,
            gen_teacher_data.generate_reasoning_prompt,
        ),
    ],
)
def test_every_prompt_template_is_fully_rendered(monkeypatch, templates, generator):
    original_choice = gen_teacher_data.random.choice

    for template in templates:
        monkeypatch.setattr(
            gen_teacher_data.random,
            "choice",
            lambda options, selected=template: (
                selected if options is templates else original_choice(options)
            ),
        )
        prompt = generator()
        assert "{" not in prompt
        assert "}" not in prompt


def test_teacher_generation_requires_provider_sdk(monkeypatch):
    monkeypatch.setattr(gen_teacher_data, "OPENAI_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="OpenAI SDK"):
        gen_teacher_data.query_teacher_model("Explain testing")


def test_teacher_generation_does_not_serialize_provider_errors(monkeypatch):
    failing_completions = SimpleNamespace(
        create=lambda **kwargs: (_ for _ in ()).throw(ConnectionError("offline"))
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=failing_completions))
    monkeypatch.setattr(gen_teacher_data, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(gen_teacher_data, "OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="Teacher model request failed"):
        gen_teacher_data.query_teacher_model("Explain testing")


def test_teacher_dataset_aborts_on_generation_error(monkeypatch, tmp_path):
    output_path = tmp_path / "teacher.jsonl"
    monkeypatch.setattr(
        gen_teacher_data,
        "generate_example",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        gen_teacher_data.generate_dataset(1, str(output_path))

    assert output_path.read_text() == ""


def test_preference_generation_requires_provider_sdk(monkeypatch):
    monkeypatch.setattr(gen_preference_data, "OPENAI_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="OpenAI SDK"):
        gen_preference_data.generate_candidate_responses("Explain testing")


def test_preference_ranking_rejects_invalid_provider_labels():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not a ranking"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )

    with pytest.raises(RuntimeError, match="invalid permutation"):
        gen_preference_data.rank_responses(
            "Explain testing",
            ["response one", "response two"],
            client=client,
        )


def test_preference_dataset_aborts_on_generation_error(monkeypatch, tmp_path):
    output_path = tmp_path / "preferences.jsonl"
    monkeypatch.setattr(
        gen_preference_data,
        "generate_preference_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        gen_preference_data.generate_preference_dataset(1, str(output_path))

    assert output_path.read_text() == ""
