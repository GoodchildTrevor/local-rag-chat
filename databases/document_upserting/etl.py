"""
Legacy ETL entry point — now delegates to IngesterClient.

Keep this file so existing callers / cron jobs that run
`python -m databases.document_upserting.etl` continue to work.
"""
import asyncio
import logging
import os
from pathlib import Path

from databases.ingestion.client import IngesterClient
from config.consts.database import FILE_FORMATS

LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "qdrant_ingest.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


async def main() -> None:
    ingester_url = os.environ["QDRANT_INGESTER_URL"]
    folder_path = Path(__file__).parent.parent / "documents"

    logger.info("Starting ETL: folder=%s, ingester=%s", folder_path, ingester_url)

    client = IngesterClient(base_url=ingester_url)
    totals = await client.ingest_folder(
        folder_path=folder_path,
        allowed_formats=set(FILE_FORMATS),
    )

    for collection, count in totals.items():
        logger.info("Collection '%s': %d chunks upserted", collection, count)

    logger.info("ETL completed.")


if __name__ == "__main__":
    asyncio.run(main())
