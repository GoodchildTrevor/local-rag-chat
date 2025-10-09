from datetime import datetime
from logging import Logger
import os
from pathlib import Path
import pytz
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Generator, Optional, Type
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types as types


def parse_creation_time(
        logger: Logger,
        snapshot: Optional[types.SnapshotDescription]
    ) -> Optional[datetime]:
    """
    Parse the creation time of a Qdrant snapshot.
    :param logger: Logger instance for logging events.
    :param snapshot: The snapshot object from Qdrant.
    :return: Naive datetime object (without timezone), or None if parsing fails.
    """
    try:
        timestamp_str = snapshot.creation_time.replace("Z", "+00:00")
        dt = datetime.fromisoformat(timestamp_str)
        return dt.replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"⚠️ Could not parse date in {snapshot.name}: {e}")
        return None


def filter_old_snapshots(
        logger: Logger,
        snapshots: list[Optional[types.SnapshotDescription]], 
        cutoff: datetime
    ) -> Generator[Optional[types.SnapshotDescription], None, None]:
    """
    Yield snapshots that were created before the cutoff date.
    :param logger: Logger instance for logging events.
    :param snapshots: List of snapshot objects.
    :param cutoff: The threshold date; snapshots older than this will be yielded.
    :yield: Snapshots created before cutoff.
    """
    for snapshot in snapshots:
        created_at = parse_creation_time(logger, snapshot)
        if created_at and created_at < cutoff:
            yield snapshot


def filter_today_snapshots(
        logger: Logger,
        snapshots: list[Optional[types.SnapshotDescription]], 
        today: datetime
    ) -> Generator[Optional[types.SnapshotDescription], None, None]:
    """
    Yield snapshots created today (same date, ignoring time).
    :param logger: Logger instance for logging events.
    :param snapshots: List of snapshot objects.
    :param today: Reference date (usually current day).
    :yield: Snapshots created on the same calendar day as `today`.
    """
    today_date = today.date()
    for snapshot in snapshots:
        created_at = parse_creation_time(logger, snapshot)
        if created_at and created_at.date() == today_date:
            yield snapshot


def delete_snapshot(
        logger: Logger,
        client: QdrantClient,
        collection: str, 
        snapshot: Optional[types.SnapshotDescription]
    ) -> None:
    """
    Delete a snapshot from the specified collection.
    :param logger: Logger instance for logging events.
    :param client: Qdrant client instance.
    :param collection: Name of the Qdrant collection.
    :param snapshot: Snapshot to delete.
    """
    try:
        client.delete_snapshot(collection_name=collection, snapshot_name=snapshot.name)
        created_at = parse_creation_time(logger, snapshot)
        time_str = created_at.strftime('%Y-%m-%d %H:%M') if created_at else "unknown"
        logger.info(f"🗑️ Deleted '{snapshot.name}' (created at: {time_str}) from '{collection}'")
    except Exception as e:
        logger.error(f"❌ Failed to delete snapshot '{snapshot.name}' from '{collection}': {e}")


def create_snapshot(
    logger: Logger,
    client: QdrantClient,
    collection: str,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    retry_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> str:
    """
    Create a new snapshot for the specified collection with retry logic.
    :param logger: Logger instance for logging events.
    :param client: Qdrant client instance.
    :param collection: Name of the Qdrant collection.
    :param max_retries: Maximum number of retry attempts (default: 3).
    :param retry_delay: Initial delay between retries in seconds (exponential backoff).
    :param retry_exceptions: Tuple of exception types to retry (default: all exceptions).
    :return: snapshot's name if successful, empty string otherwise.
    """
    if retry_exceptions is None:
        retry_exceptions = (Exception,)

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=retry_delay),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True,
    )
    def _create_snapshot_attempt():
        snapshot = client.create_snapshot(collection_name=collection)
        logger.info(f"✅ Created snapshot '{snapshot.name}' for collection '{collection}'")
        return snapshot.name

    try:
        return _create_snapshot_attempt()
    except retry_exceptions as e:
        logger.error(f"❌ Failed to create snapshot after {max_retries} attempts: {e}")


def download_snapshot(
    logger: Logger,
    backup_dir: str,
    qdrant_url: str,
    collection_name: str,
    snapshot_name: str,
) -> Path:
    """
    Download a Qdrant snapshot from a local/remote instance
    :param logger: Logger instance for logging events.
    :param collection_name: Name of the Qdrant collection.
    :param snapshot_name: Name of the snapshot to download.
    :return: Path to the downloaded snapshot file.
    :raises requests.HTTPError: If download fails.
    :raises OSError: If file cannot be saved.
    """
    os.makedirs(backup_dir, exist_ok=True)
    snapshot_url = f"{qdrant_url.rstrip('/')}/collections/{collection_name}/snapshots/{snapshot_name}"
    filepath = Path(backup_dir) / f"{snapshot_name}.snapshot"

    if logger:
        logger.info(f"⏬ Downloading snapshot from {snapshot_url}...")

    try:
        response = requests.get(snapshot_url, stream=True)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if logger:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Snapshot saved: {filepath} ({size_mb:.1f} MB)")

        return filepath

    except requests.HTTPError as e:
        error_msg = f"HTTP error: {e.response.status_code}"
        if logger:
            logger.error(error_msg)
        raise
    except OSError as e:
        error_msg = f"Filesystem error: {e}"
        if logger:
            logger.error(error_msg)
        raise
    

def get_latest_snapshot(
        logger: Logger, 
        collection_name: str, 
        snapshots: list[str]
    ) -> str:
    """
    :param logger: Logger instance for logging events.
    :param collection_name: Name of the Qdrant collection.
    :
    :return: name of the latest snapshot
    """
    snapshot_list = list()
    for snapshot in snapshots:
        created_str = snapshot.creation_time 
        if created_str:
            created_time = datetime.strptime(created_str, "%Y-%m-%dT%H:%M:%S")
            created_time = pytz.UTC.localize(created_time)
        else:
            logger.error("Not possible to find latest snapshot")

        snapshot_list.append((snapshot, created_time))
    
    latest_snapshot_info = max(snapshot_list, key=lambda x: x[1])
    latest_snapshot_name = latest_snapshot_info[0].name
    logger.info(f"Latest snapshot for {collection_name} is {latest_snapshot_name}")

    return latest_snapshot_name
