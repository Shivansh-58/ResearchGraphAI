"""Download public weights into the local model server during connected setup."""

import argparse

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b", "nomic-embed-text"])
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=1200, trust_env=False) as client:
        for model in args.models:
            print(f"Preparing {model} locally…", flush=True)
            response = client.post("/api/pull", json={"model": model, "stream": False})
            response.raise_for_status()
            if response.json().get("status") != "success":
                raise RuntimeError("Model preparation failed.")
            print(f"Prepared {model}.", flush=True)


if __name__ == "__main__":
    main()
