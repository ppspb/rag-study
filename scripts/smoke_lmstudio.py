from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    model = os.getenv("LM_STUDIO_EMBED_MODEL")
    if not model:
        raise SystemExit("Set LM_STUDIO_EMBED_MODEL in .env")

    client = OpenAI(base_url=base_url, api_key="lm-studio")
    resp = client.embeddings.create(
        model=model,
        input=[
            "What is BM25 in information retrieval?",
            "How do qrels affect evaluation?",
        ],
    )
    dims = len(resp.data[0].embedding)
    print({"base_url": base_url, "model": model, "vectors": len(resp.data), "dims": dims})


if __name__ == "__main__":
    main()
