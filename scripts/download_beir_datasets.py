from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

import yaml
from beir import util


@dataclass
class DownloadRecord:
    dataset: str
    split: str
    url: str
    extracted_path: str
    exists_after_download: bool


def load_config(config_path: pathlib.Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "configs" / "datasets.yaml")

    base_data_dir = repo_root / config["base_data_dir"]
    output_dir = repo_root / config["output_dir"]
    base_data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[DownloadRecord] = []

    for ds in config["datasets"]:
        if not ds.get("enabled", True):
            continue

        dataset_name = ds["name"]
        split = ds.get("split", "test")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        extracted = util.download_and_unzip(url, str(base_data_dir))
        records.append(
            DownloadRecord(
                dataset=dataset_name,
                split=split,
                url=url,
                extracted_path=str(extracted),
                exists_after_download=pathlib.Path(extracted).exists(),
            )
        )

    manifest_path = output_dir / "download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
