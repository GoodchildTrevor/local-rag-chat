from datetime import datetime, timedelta
import logging

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from databases.snapshots.snapshot_utils import (
    get_latest_snapshot,
)
from config.settings import AppConfig, ClientsConfig, DBConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

app_config = AppConfig()
client_config = ClientsConfig()
db_config = DBConfig()

BACKUP_DIR = os.path.join(os.path.dirname(__file__), db_config.rag_snapshot_dir)


def recover_snapshots(
        app_config: AppConfig,
        client_config: ClientsConfig,
        collections_snapshots_names: dict[str, None] = None,
    ) -> None:
    """
    Recovers Qdrant collection snapshots from a configured URL.
    :param app_config: Configuration object containing collection attributes.
    :param client_config: Configuration object containing the Qdrant client and URL.
    :param collections_snapshots_names: Optional dictionary mapping collection names
    to specific snapshot names to restore. If a collection is not in this dictionary,
    it will be skipped. If a value is ``None``, the latest snapshot will be used.
    :raises Exception: If snapshot recovery fails for any collection, an error is
    logged but the exception is caught and does not halt execution for other collections.
    """
    client = client_config.qdrant_client
    qdrant_url = client_config.qdrant_url
    attrs = [attr for attr in dir(app_config) if "collection" in attr]
    collections = [getattr(app_config, collection) for collection in attrs]

    if not collections:
        logger.error("❌ No collections found in configuration")
        return
    
    filter_collections = dict()
    for coll in collections:
        if coll in collections_snapshots_names.keys():
            filter_collections[coll] = collections_snapshots_names[coll]
        else:
            logger.warning(f"Collections {coll} not in existing collections")
    
    for coll, snapshot_name in filter_collections.items():
        try:
            snapshots = client.list_snapshots(collection_name=coll)
            logger.info(f"📦 Found {len(snapshots)} snapshots for collection '{coll}'")
            if snapshot_name:
                if snapshot_name in snapshots:
                    filter_snapshot_name = snapshot_name
                else:
                    logger.error(f"Snapshot {snapshot_name} not in snaphots")
            else:
                filter_snapshot_name = get_latest_snapshot(logger, coll, snapshots)
            snapshot_url = f"{qdrant_url.rstrip('/')}/collections/{coll}/snapshots/{filter_snapshot_name}"
            client.recover_snapshot(
                collection_name=coll,
                location=snapshot_url
            )

        except Exception as e:
            logger.error(f"❌ Failed to process collection '{coll}': {e}")


if __name__ == "__main__":
    collections_snapshots_names = {
        "docs": None
    }
    recover_snapshots(
        app_config=app_config,
        client_config=client_config,
        collections_snapshots_names =collections_snapshots_names, 
    )
