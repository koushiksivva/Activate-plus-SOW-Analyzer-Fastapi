from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from utils import (
    process_pdf_safely, extract_durations_optimized, store_chunks_in_cosmos,
    process_batch_with_fallback, create_excel_with_formatting, generate_document_id,
    task_batches, normalize_and_clean_text, collection
)
import os
import logging
import tempfile
from dotenv import load_dotenv
import uuid
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Project Plan Agent")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory task status
tasks = {}  # task_id: {'status': 'processing'|'completed'|'failed', 'excel_path': str or None, 'error': str or None}

class UploadResponse(BaseModel):
    status: str
    task_id: str

class StatusResponse(BaseModel):
    status: str
    download_url: str or None = None
    error: str or None = None

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load frontend")

async def process_pdf_background(task_id: str, file_path: str, filename: str):
    logger.info(f"Starting background processing for task_id: {task_id}, file: {filename}")
    global tasks
    try:
        logger.info(f"Checking file existence: {os.path.exists(file_path)}")
        logger.info(f"Setting task {task_id} to processing")
        tasks[task_id] = {'status': 'processing', 'excel_path': None, 'error': None}

        processing_result = process_pdf_safely(file_path)
        if processing_result is None:
            raise ValueError("No readable content found in the PDF")

        pdf_text, normalized_pdf_text, tmp_pdf_path, images_content = processing_result

        logger.info("Extracting phase durations...")
        durations = extract_durations_optimized(pdf_text)

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        chunks = splitter.split_text(pdf_text) if pdf_text.strip() else []

        if not chunks:
            raise ValueError("No valid text content to process")

        document_id = generate_document_id(pdf_text)
        logger.info(f"Processing document with ID: {document_id}")

        logger.info("Storing document chunks...")
        success = store_chunks_in_cosmos(chunks, images_content, document_id)
        if not success:
            raise ValueError("Failed to store document in Cosmos DB")

        stored_count = collection.count_documents({"document_id": document_id})
        logger.info(f"Stored {stored_count} chunks in database")

        logger.info("Analyzing tasks in SOW...")
        all_tasks = []
        for heading, tasks in task_batches.items():
            if heading and tasks:
                for task in tasks:
                    if task and task.strip():
                        all_tasks.append((str(heading), str(task)))

        if not all_tasks:
            raise ValueError("No valid tasks found to process")

        batch_size = 15
        task_batches_split = [all_tasks[i:i+batch_size] for i in range(0, len(all_tasks), batch_size)]

        from functools import partial
        from concurrent.futures import ThreadPoolExecutor
        process_fn = partial(
            process_batch_with_fallback,
            document_id=document_id,
            durations=durations,
            normalized_pdf_text=normalized_pdf_text,
            pdf_text=pdf_text
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(process_fn, task_batches_split))

        flat_rows = []
        for batch_result in results:
            if batch_result and isinstance(batch_result, list):
                flat_rows.extend(batch_result)

        if not flat_rows:
            raise ValueError("Failed to process any tasks")

        import pandas as pd
        df = pd.DataFrame(flat_rows)
        df = df[df['Present'] != 'error']
        if df.empty:
            raise ValueError("All tasks failed processing")

        tmp_excel_path = f"/tmp/{task_id}.xlsx"
        create_excel_with_formatting(df, durations, tmp_excel_path, activity_column_width=50)

        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.unlink(tmp_pdf_path)

        logger.info(f"Setting task {task_id} to completed")
        tasks[task_id] = {'status': 'completed', 'excel_path': tmp_excel_path, 'error': None}
        logger.info(f"Processing completed for task_id: {task_id}")

    except Exception as e:
        logger.error(f"Error in background processing for task_id: {task_id}: {str(e)}")
        tasks[task_id] = {'status': 'failed', 'excel_path': None, 'error': str(e)}
        if os.path.exists(file_path):
            os.unlink(file_path)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Process uploaded PDF in background and return task ID"""
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        task_id = str(uuid.uuid4())
        file_path = f"/tmp/{task_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        background_tasks.add_task(process_pdf_background, task_id, file_path, file.filename)

        return UploadResponse(status="accepted", task_id=task_id)

    except Exception as e:
        logger.error(f"Error initiating upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate PDF processing: {str(e)}")

@app.get("/status/{task_id}", response_model=StatusResponse)
async def check_status(task_id: str):
    """Check status of a processing task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task['status'] == 'completed':
        download_url = f"/download/{task_id}"
    else:
        download_url = None

    return StatusResponse(status=task['status'], download_url=download_url, error=task['error'])

@app.get("/download/{task_id}")
async def download_excel(task_id: str):
    """Download the generated Excel file"""
    if task_id not in tasks or tasks[task_id]['status'] != 'completed':
        raise HTTPException(status_code=404, detail="File not ready or not found")

    excel_path = tasks[task_id]['excel_path']
    if not os.path.exists(excel_path):
        raise HTTPException(status_code=404, detail="File not found")

    response = FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="AI-Generated_SOW_Document.xlsx",
        background=BackgroundTask(lambda: os.unlink(excel_path) if os.path.exists(excel_path) else None)
    )

    del tasks[task_id]

    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
