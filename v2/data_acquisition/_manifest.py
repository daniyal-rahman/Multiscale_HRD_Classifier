"""
Shared manifest tracking for data acquisition scripts.

Each download script calls update_manifest() to record what was downloaded and when.
The manifest is saved as manifest.json in the data_acquisition directory.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def load_manifest():
    """Load the current manifest, or return an empty one."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"datasets": {}, "last_updated": None}


def update_manifest(dataset_key, info):
    """Update the manifest with info about a downloaded dataset.

    Parameters
    ----------
    dataset_key : str
        Identifier for the dataset (e.g., 'tcga', 'geo', 'gdsc_ccle').
    info : dict
        Metadata about what was downloaded.
    """
    manifest = load_manifest()

    manifest["datasets"][dataset_key] = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        **info,
    }
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Updated manifest for '%s'", dataset_key)


def get_dataset_status(dataset_key=None):
    """Check the download status of a dataset (or all datasets)."""
    manifest = load_manifest()

    if dataset_key:
        return manifest["datasets"].get(dataset_key)

    return manifest["datasets"]
