import json
import shutil
from pathlib import Path
from typing import Any

import click
from kaggle.api.kaggle_api_extended import KaggleApi


def copy_files_with_exts(source_dir: Path, dest_dir: Path, exts: list):
    for ext in exts:
        for source_path in source_dir.rglob(f"*{ext}"):
            relative_path = source_path.relative_to(source_dir)
            dest_path = dest_dir / relative_path

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")


@click.command()
@click.option("--title", "-t", default="CIBMTR-model")
@click.option("--dirs", "-d", type=list[Path], default=[Path("./model"), Path("./processed_data")])
@click.option("--extentions", "-e", type=list[str], default=["exp*best_model*.pth", "v*encoder.pkl", "v*scaler.pkl", "exp*_processed_data.yaml"])
@click.option("--user_name", "-u", default=None)
@click.option("--new", "-n", is_flag=True)
def main(
    title: str,
    dirs: list[Path],
    extentions: list[str] = [".pth", ".yaml"],
    user_name: str = None,
    new: bool = False,
):
    for dir in dirs:
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        copy_files_with_exts(dir, tmp_dir, extentions)

    dataset_metadata: dict[str, Any] = {}
    dataset_metadata["id"] = f"{user_name}/{title}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = title
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)

    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )

    # delete tmp dir
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()