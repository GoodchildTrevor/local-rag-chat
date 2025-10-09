from datetime import datetime, timedelta
import logging

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from databases.snapshots.snapshot_utils import (
    create_snapshot,
    delete_snapshot,
    download_snapshot,
    filter_old_snapshots,
    filter_today_snapshots,
)
from config.settings import AppConfig, ClientsConfig, DBConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

app_config = AppConfig()
client_config = ClientsConfig()
db_config = DBConfig()

BACKUP_DIR = os.path.join(os.path.dirname(__file__), db_config.rag_snapshot_dir)


def refresh_backups(
        days: int,
        app_config: AppConfig,
        client_config: ClientsConfig
    ) -> None:
    """
    Manage Qdrant snapshots: delete old ones and create a new one if not exists today.
    This function:
      - Lists all snapshots for each collection.
      - Deletes snapshots older than threshold.
      - Creates a new snapshot if no snapshot exists for today.
    :param days: Age threshold in days for snapshot deletion.
    :param app_config: base configs with collection names
    :param client_config: configs of QdrantClient for connection and requests
    """
    client = client_config.qdrant_client
    qdrant_url = client_config.qdrant_url
    attrs = [attr for attr in dir(app_config) if "collection" in attr]
    collections = [getattr(app_config, collection) for collection in attrs]

    now = datetime.now()
    cutoff = now - timedelta(days=days)

    if not collections:
        logger.error("❌ No collections found in configuration")
        return

    for coll in collections:
        try:
            snapshots = client.list_snapshots(collection_name=coll)
            logger.info(f"📦 Found {len(snapshots)} snapshots for collection '{coll}'")

            old_snapshots = list(filter_old_snapshots(logger, snapshots, cutoff))
            if old_snapshots:
                logger.info(
                    f"🧹 Deleting {len(old_snapshots)} snapshots older than {days} days ({cutoff.strftime('%Y-%m-%d %H:%M')})"
                    )
                for snapshot in old_snapshots:
                    delete_snapshot(logger, client, coll, snapshot)
            else:
                logger.debug(f"✅ No old snapshots to delete in '{coll}'")

            if not any(filter_today_snapshots(logger, snapshots, now)):
                snapshot_name = create_snapshot(logger, client, coll)
                if snapshot_name:
                    download_snapshot(
                        logger=logger, 
                        backup_dir=BACKUP_DIR, 
                        qdrant_url=qdrant_url,
                        collection_name=coll, 
                        snapshot_name=snapshot_name
                    )
            else:
                logger.warning(f"🟡 Skipping creation: a snapshot already exists for today in '{coll}'")

        except Exception as e:
            logger.error(f"❌ Failed to process collection '{coll}': {e}")


if __name__ == "__main__":
    refresh_backups(
        days=30,
        app_config=app_config,
        client_config=client_config,
    )
