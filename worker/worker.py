import os
import json
import time
import logging
from datetime import datetime
from azure.storage.queue import QueueClient
from utils import process_job_message, jobs_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("worker")

AZURE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
QUEUE_NAME = os.getenv("STORAGE_QUEUE_NAME", "sow-jobs")
UPLOAD_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER_UPLOADS", "uploads")
RESULTS_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER_RESULTS", "results")

queue_client = QueueClient.from_connection_string(AZURE_CONN, QUEUE_NAME)

def handle_message(msg):
    try:
        body = json.loads(msg.content)
    except Exception:
        logger.exception("Invalid message, deleting")
        queue_client.delete_message(msg.id, msg.pop_receipt)
        return

    job_id = body.get("job_id")
    blob_name = body.get("blob_name")
    if not job_id or not blob_name:
        logger.warning("Message missing job_id or blob_name")
        queue_client.delete_message(msg.id, msg.pop_receipt)
        return

    job_doc = jobs_collection.find_one({"job_id": job_id}) or {}
    attempts = job_doc.get("attempts", 0)

    if attempts >= int(os.getenv("MAX_JOB_ATTEMPTS", "3")):
        jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "failed",
                "error": "max attempts exceeded",
                "updated_at": datetime.utcnow()
            }}
        )
        queue_client.delete_message(msg.id, msg.pop_receipt)
        logger.info("Deleted message for job %s after %s attempts", job_id, attempts)
        return

    # Try processing the job
    ok = process_job_message(
        job_id=job_id,
        blob_name=blob_name,
        storage_conn_str=AZURE_CONN,
        upload_container=UPLOAD_CONTAINER,
        results_container=RESULTS_CONTAINER,
    )

    if ok:
        queue_client.delete_message(msg.id, msg.pop_receipt)
        logger.info("✅ Job %s processed successfully", job_id)
    else:
        jobs_collection.update_one(
            {"job_id": job_id},
            {"$inc": {"attempts": 1}, "$set": {"updated_at": datetime.utcnow()}}
        )
        logger.warning("❌ Job %s failed, will retry later", job_id)


if __name__ == "__main__":
    logger.info("Worker started, polling queue...")
    while True:
        messages = queue_client.receive_messages(messages_per_page=5, visibility_timeout=300)
        found = False
        for msg in messages:
            found = True
            handle_message(msg)
        if not found:
            time.sleep(5)
