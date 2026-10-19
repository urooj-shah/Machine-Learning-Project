"""Download Kvasir-SEG dataset from Kaggle to data/raw/."""

from pathlib import Path
import shutil

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "kvasir-seg"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Kvasir-SEG from Kaggle ...")
    path = kagglehub.dataset_download("debeshjha1/kvasirseg")
    print(f"Kaggle cached at: {path}")

    if Path(path).resolve() != DATA_DIR.resolve():
        print(f"Copying to {DATA_DIR} ...")
        shutil.copytree(path, DATA_DIR, dirs_exist_ok=True)

    print(f"Done. Dataset at: {DATA_DIR}")


if __name__ == "__main__":
    main()
