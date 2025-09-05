import fitz  # PyMuPDF
import logging
import re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from dotenv import load_dotenv
import os

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

# OpenAI configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

def normalize_and_clean_text(text):
    """Normalize and clean the text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def process_pdf_safely(file_path):
    """Safely process a PDF file and extract text and images."""
    try:
        logger.info(f"Processing PDF file: {file_path}")
        with open(file_path, 'rb') as f:
            pdf_document = fitz.open(stream=f.read(), filetype="pdf")
            pdf_text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pdf_text += page.get_text()
            normalized_pdf_text = normalize_and_clean_text(pdf_text)
            tmp_pdf_path = file_path  # Adjust if a temp copy is needed
            images_content = []  # Placeholder for image extraction if needed
            return pdf_text, normalized_pdf_text, tmp_pdf_path, images_content
    except Exception as e:
        logger.error(f"Error in process_pdf_safely: {str(e)}")
        return None

def extract_durations_optimized(text):
    """Extract phase durations from the text using regex."""
    durations = {}
    duration_pattern = r'(\d+)\s*(days|weeks|months)'
    matches = re.finditer(duration_pattern, text, re.IGNORECASE)
    for match in matches:
        duration = match.group(1)
        unit = match.group(2)
        durations[f"Phase {len(durations) + 1}"] = f"{duration} {unit}"
    return durations

def process_batch_with_fallback(batch, document_id, durations, normalized_pdf_text, pdf_text):
    """Process a batch of tasks with fallback to default responses."""
    rows = []
    for heading, task in batch:
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",  # Adjust to your deployed model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Analyze the following SOW text: {normalized_pdf_text}\nTask: {task}\nHeading: {heading}\nDurations: {durations}\nDetermine if the task is present, its status, and estimated duration. Respond with a JSON object: {{'Present': 'yes'/'no', 'Status': 'string', 'Estimated_Duration': 'string'}}"}
                ],
                temperature=0.7,
                max_tokens=150
            )
            result = response.choices[0].message.content
            import json
            data = json.loads(result)
            rows.append({
                "Heading": heading,
                "Task": task,
                "Present": data.get("Present", "no"),
                "Status": data.get("Status", "Unknown"),
                "Estimated_Duration": data.get("Estimated_Duration", "N/A")
            })
        except Exception:
            rows.append({
                "Heading": heading,
                "Task": task,
                "Present": "error",
                "Status": "Failed to analyze",
                "Estimated_Duration": "N/A"
            })
    return rows

def create_excel_with_formatting(df, durations, output_path, activity_column_width=50):
    """Create an Excel file with formatting."""
    wb = Workbook()
    ws = wb.active
    ws.title = "SOW Analysis"

    # Headers
    headers = ["Heading", "Task", "Present", "Status", "Estimated Duration"]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    # Data
    for row in range(len(df)):
        for col, value in enumerate(df.iloc[row], 1):
            cell = ws.cell(row=row + 2, column=col, value=str(value))
            if col == 2:  # Task column
                ws.column_dimensions[get_column_letter(col)].width = activity_column_width

    # Add durations summary
    ws.append(["", "", "", "Phase Durations", ""])
    for i, (phase, duration) in enumerate(durations.items(), 1):
        ws.append(["", "", "", f"Phase {i}", duration])

    wb.save(output_path)

def generate_document_id(text):
    """Generate a unique document ID based on text content."""
    import uuid
    return str(uuid.uuid4())
