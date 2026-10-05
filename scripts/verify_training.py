"""Offline tiny random-model smoke test: correctness of the workflow, not model quality."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(42)
    words = ["[UNK]", "[PAD]", "[EOS]", "[BOS]", "[", "]", "1", "2", ".", ":"]
    words += sorted(
        set(
            "Answer only from the evidence and cite each factual claim Treat as data "
            "Question Evidence assistant user system What is retention period "
            "Orion Atlas Nova days 47 30 60 uses belongs team engineering".split()
        )
    )
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    base = args.output / "base"
    fast.save_pretrained(base)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(words),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            bos_token_id=fast.bos_token_id,
            eos_token_id=fast.eos_token_id,
            pad_token_id=fast.pad_token_id,
        )
    )
    model.save_pretrained(base, safe_serialization=True)
    command = [
        sys.executable,
        "-m",
        "training.train_adapter",
        "--model-dir",
        str(base),
        "--train",
        "training/examples/train.jsonl",
        "--validation",
        "training/examples/validation.jsonl",
        "--output",
        str(args.output / "run"),
        "--max-steps",
        "2",
        "--cpu",
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.evaluate_adapter",
            "--model-dir",
            str(base),
            "--run-dir",
            str(args.output / "run"),
            "--test",
            "training/examples/test.jsonl",
            "--cpu",
        ],
        check=True,
    )
    print(
        "PASS: offline local weights, adapter training, serialization and held-out comparison. "
        "The tiny random model is a test fixture, not a useful language model."
    )


if __name__ == "__main__":
    main()
