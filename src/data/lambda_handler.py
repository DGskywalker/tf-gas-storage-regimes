"""AWS Lambda handler for scheduled daily GIE AGSI+ storage data ingestion to Amazon S3."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import boto3

from src.data.gie_client import GIEClient

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler invoked via Amazon EventBridge on a daily cron schedule.

    Fetches the latest D-1 European storage inventory data from GIE AGSI+ API
    and persists the raw JSON/Parquet payload to the configured S3 bucket.
    """
    s3_bucket = os.environ.get("STORAGE_S3_BUCKET", "ttf-gas-storage-lake")
    api_key = os.environ.get("GIE_API_KEY", "")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("Starting daily GIE AGSI+ ingestion for %s into s3://%s", today_str, s3_bucket)

    try:
        client = GIEClient(api_key=api_key)
        # Fetch latest aggregate EU storage fill
        storage_records = client.fetch_historical_range(
            country="EU",
            start_date=today_str,
            end_date=today_str,
        )

        s3 = boto3.client("s3")
        s3_key = f"raw/gie_agsi/year={today_str[:4]}/month={today_str[5:7]}/eu_storage_{today_str}.json"

        payload = json.dumps([r.model_dump() for r in storage_records], default=str)
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )

        logger.info("Successfully ingested %d records to s3://%s/%s", len(storage_records), s3_bucket, s3_key)
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "date": today_str,
                "records_ingested": len(storage_records),
                "s3_uri": f"s3://{s3_bucket}/{s3_key}",
            }),
        }
    except Exception as exc:
        logger.error("Failed to ingest GIE storage data: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "message": str(exc)}),
        }
