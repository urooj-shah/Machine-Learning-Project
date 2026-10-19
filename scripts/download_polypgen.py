"""Download PolypGen dataset from Synapse to data/raw/."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import synapseclient

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SYNAPSE_ENTITY = "syn45200214"


def main():
    load_dotenv(DATA_DIR.parent.parent / ".env")
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        print("Set SYNAPSE_AUTH_TOKEN env var before running.")
        print("  export SYNAPSE_AUTH_TOKEN='your-token-here'")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    print(f"Downloading {SYNAPSE_ENTITY} to {DATA_DIR} ...")
    entity = syn.get(entity=SYNAPSE_ENTITY, downloadLocation=str(DATA_DIR))
    print(f"Done. File at: {entity.path}")


if __name__ == "__main__":
    main()
