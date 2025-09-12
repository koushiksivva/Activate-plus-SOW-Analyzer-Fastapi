from fastapi import FastAPI, UploadFile, File, HTTPException
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
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Project Plan Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load frontend")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Process PDF asynchronously
        loop = asyncio.get_event_loop()
        processing_result = await loop.run_in_executor(None, lambda: process_pdf_safely(file))
        if processing_result is None:
            raise HTTPException(status_code=400, detail="No readable content found in the PDF")

        pdf_text, normalized_pdf_text, tmp_pdf_path, images_content = processing_result
        durations = await loop.run_in_executor(None, lambda: extract_durations_optimized(pdf_text))
        logger.info("Extracting phase durations...")

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        chunks = splitter.split_text(pdf_text) if pdf_text.strip() else []

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text content to process")

        document_id = generate_document_id(pdf_text)
        logger.info(f"Processing document with ID: {document_id}")

        # Store chunks in Cosmos DB
        success = await loop.run_in_executor(None, lambda: store_chunks_in_cosmos(chunks, images_content, document_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document in Cosmos DB")

        stored_count = collection.count_documents({"document_id": document_id})
        logger.info(f"Stored {stored_count} chunks in database")

        # Process tasks
        logger.info("Analyzing tasks in SOW...")
        all_tasks = [(str(heading), str(task)) for heading, tasks in task_batches.items() for task in tasks if task and task.strip()]
        if not all_tasks:
            raise HTTPException(status_code=400, detail="No valid tasks found to process")

        batch_size = 15
        task_batches_split = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]

        # Sequentially process batches one by one
        results = []
        for idx, batch in enumerate(task_batches_split):
            logger.info(f"Processing batch {idx + 1} of {len(task_batches_split)}")
            result = await process_batch(batch)
            if result:
                results.append(result)

        flat_rows = [row for result in results for row in result if result and isinstance(result, list)]

        if not flat_rows:
            raise HTTPException(status_code=500, detail="Failed to process any tasks")

        import pandas as pd
        df = pd.DataFrame(flat_rows)
        df = df[df['Present'] != 'error']
        if df.empty:
            raise HTTPException(status_code=500, detail="All tasks failed processing")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_excel:
            await loop.run_in_executor(None, lambda: create_excel_with_formatting(df, durations, tmp_excel.name, activity_column_width=50))
            tmp_excel_path = tmp_excel.name

        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.unlink(tmp_pdf_path)

        response = FileResponse(
            tmp_excel_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="AI-Generated_SOW_Document.xlsx",
            background=BackgroundTask(lambda: os.unlink(tmp_excel_path) if os.path.exists(tmp_excel_path) else None)
        )
        return response

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
