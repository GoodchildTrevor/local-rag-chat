import logging
from pathlib import Path
import os
import requests
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.settings import ClientsConfig, DBConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

client_config = ClientsConfig()
db_config = DBConfig()

qdrant_url = client_config.qdrant_url
BACKUP_DIR = Path(os.path.join(os.path.dirname(__file__), db_config.rag_snapshot_dir))

def restore_snapshots(
        backup_dir: str,
        qdrant_url: str
    ) -> None:
    """
    Restores Qdrant collection snapshots from a local backup directory.
    This function walks through the specified backup directory, finds all files with 
    the `.snapshot` extension, and uploads them to the Qdrant instance to restore 
    collections (e.g., in case of data loss). The target collection is assumed to be 
    'test_collection_import' — adjust if needed.
    :param backup_dir: Path to the directory containing snapshot files.
    :param qdrant_url: Base URL of the Qdrant service.
    :raises Exception: If snapshot upload fails (non-200 response).
    """
    for root, dirs, files in os.walk(backup_dir):
        for snapshot_name in files:
            logger.info(f"Processing snapshot {snapshot_name}")
            if not snapshot_name.endswith(".snapshot"):
                continue

            snapshot_path = os.path.join(root, snapshot_name) 

            with open(snapshot_path, "rb") as f:
                response = requests.post(
                    f"{qdrant_url}/collections/test_collection_import/snapshots/upload",
                    params={"priority": "snapshot"},
                    files={"snapshot": (snapshot_name, f, "application/octet-stream")},
                )

            if response.status_code == 200:
                logger.info(f"✅ Snapshot restored successfully: {snapshot_name}")
            else:
                raise logger.error(f"❌ Snapshot restoring failed {snapshot_name}: {response.status_code} {response.text}")
            

if __name__=="__main__":
    restore_snapshots(BACKUP_DIR, qdrant_url)
            