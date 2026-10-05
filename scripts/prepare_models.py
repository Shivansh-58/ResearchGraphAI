"""Connected preparation only; never used during private inference or training."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--revision", required=True, help="Pin a model revision/commit for reproducibility."
    )
    parser.add_argument("--output", type=Path, default=Path("models/base"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(
        args.model,
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "tokenizer.*",
            "*.txt",
            "*.model",
            "README.md",
            "LICENSE*",
        ],
        ignore_patterns=["*.bin"],
    )
    print(f"Weights prepared in {args.output}. Runtime uses local_files_only=True.")


if __name__ == "__main__":
    main()
