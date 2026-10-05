from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from training.data import dataset_hash, load_examples, question_hash, score_answer
from training.train_adapter import messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local base and adapter on held-out examples."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--approve-for-serving",
        action="store_true",
        help="Explicit approval after reviewing held-out quality and privacy.",
    )
    args = parser.parse_args()
    examples = load_examples(args.test)
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_hash(args.model_dir / "config.json") != manifest.get("base_config_sha256"):
        raise ValueError("Evaluation must use the same base-model configuration as training.")
    if dataset_hash(args.test) in {manifest["train_sha256"], manifest["validation_sha256"]}:
        raise ValueError("Use a separate test file.")
    if {question_hash(item) for item in examples} & set(manifest.get("seen_question_hashes", [])):
        raise ValueError("Test questions overlap with training or validation.")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(base, args.run_dir / "adapter", local_files_only=True)
    model.eval()
    results = {"base": [], "adapter": []}
    for example in examples:
        encoded = tokenizer(
            tokenizer.apply_chat_template(
                messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        if encoded.input_ids.shape[1] > 3072:
            raise ValueError("Test context exceeds budget.")
        for variant in results:
            with torch.inference_mode():
                if variant == "base":
                    with model.disable_adapter():
                        output = model.generate(
                            **encoded,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                else:
                    output = model.generate(
                        **encoded,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            answer = tokenizer.decode(
                output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
            )
            results[variant].append(score_answer(answer, example))
    summary = {
        variant: {metric: sum(item[metric] for item in items) / len(items) for metric in items[0]}
        for variant, items in results.items()
    }
    report = {
        "examples": len(examples),
        "test_sha256": dataset_hash(args.test),
        "metrics": summary,
        "limitations": (
            "String-based fact recall and citation syntax only; "
            "human faithfulness review is required."
        ),
    }
    (args.run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest["status"] = "evaluated"
    manifest["approved_for_serving"] = False
    manifest["metrics"].update(
        {
            f"{variant}_{metric}": value
            for variant, values in summary.items()
            for metric, value in values.items()
        }
    )
    if args.approve_for_serving:
        candidate, baseline = summary["adapter"], summary["base"]
        if (
            candidate["required_fact_recall"] < baseline["required_fact_recall"]
            or candidate["valid_citations"] < 1
        ):
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise ValueError("Adapter failed the minimum comparison gate; it was not approved.")
        manifest["approved_for_serving"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
