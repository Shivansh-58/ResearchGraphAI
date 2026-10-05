import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("training_data", ROOT / "training/data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_example_training_splits_are_valid():
    train = module.load_examples(ROOT / "training/examples/train.jsonl")
    validation = module.load_examples(ROOT / "training/examples/validation.jsonl")
    module.validate_splits(train, validation)


def test_training_question_leakage_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        module.validate_splits([{"question": "How much?"}], [{"question": "HOW MUCH?"}])


def test_training_invalid_citations_are_rejected(test_settings):
    path = test_settings.data_dir / "invalid.jsonl"
    path.write_text(
        json.dumps({"question": "What?", "context": "SOURCE [1]: data", "answer": "Data [2]"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="citation"):
        module.load_examples(path)


def test_runtime_docker_network_is_private_and_model_port_is_not_published():
    compose = yaml.safe_load((ROOT / "deploy/compose.yaml").read_text())
    assert compose["networks"]["private"]["internal"] is True
    assert "ports" not in compose["services"]["ollama"]
    assert "ports" not in compose["services"]["api"]
    assert compose["services"]["ollama"]["environment"]["OLLAMA_NO_CLOUD"] == "1"
    assert compose["services"]["trainer"]["network_mode"] == "none"


def test_frontend_has_no_external_assets():
    for path in (ROOT / "frontend/src").rglob("*.css"):
        assert "https://" not in path.read_text(encoding="utf-8")


def test_company_data_and_training_environment_are_excluded_from_image_context():
    excluded = set((ROOT / ".dockerignore").read_text().splitlines())
    assert {"secrets", "models", "data", "training/datasets", ".env*", ".training-venv"} <= excluded
