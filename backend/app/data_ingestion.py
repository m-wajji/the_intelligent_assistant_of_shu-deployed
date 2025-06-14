import os
import fitz  # PyMuPDF
import camelot
import pdfplumber
import gc
import pandas as pd
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- CONFIGURATION ---
base_pdfs_dir = "app/shu_partitioned_db/pdfs"
base_persist_directory = "app/dbs/chroma"

# Define your partitions
partitions = [
    "shu_academic_programs_db", 
    "shu_admissions_services_db",
    "shu_events_news_db",
    "shu_financial_operational_db",
    "shu_institutional_governance_db",
    "shu_research_innovation_db",
    "shu_student_life_services_db"
]

# Create base persist directory
os.makedirs(base_persist_directory, exist_ok=True)

# --- EMBEDDING MODEL (ChatGPT Embeddings) ---
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(
    openai_api_key=openai_api_key, model="text-embedding-3-large"
)

# --- TEXT SPLITTER ---
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])

# --- DEVICE CONFIGURATION ---
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- MODELS INITIALIZATION ---
class ModelManager:
    def __init__(self):
        self.yolo = None
        self.blip_processor = None
        self.blip_model = None
        self.models_loaded = False
    
    def load_models(self):
        if not self.models_loaded:
            try:
                logger.info("Loading YOLO model...")
                yolo_weights = "yolo/yolo11s.pt"
                if not os.path.exists(yolo_weights):
                    logger.warning(f"YOLO weights not found at {yolo_weights}. Skipping image processing.")
                    return False
                
                self.yolo = YOLO(yolo_weights)
                
                logger.info("Loading BLIP model...")
                self.blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(device)
                
                self.models_loaded = True
                logger.info("Models loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                return False
        return True
    
    def cleanup(self):
        """Clean up models to free memory"""
        if self.models_loaded:
            del self.yolo, self.blip_processor, self.blip_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.models_loaded = False

model_manager = ModelManager()

def summarize_image(pil_image: Image.Image) -> str:
    """Summarize image using BLIP model with CPU optimizations"""
    if not model_manager.models_loaded:
        return "Image processing not available"
    try:
        # Resize image for faster processing on CPU
        if device == "cpu":
            # Resize large images to speed up processing
            max_size = 512
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        inputs = model_manager.blip_processor(
            images=pil_image, 
            return_tensors="pt"
        ).to(device)
        
        # CPU-specific generation parameters
        generation_params = {
            "max_length": 20,
            "num_beams": 1 if device == "cpu" else 4,  # Faster inference on CPU
            "do_sample": False,  # Deterministic output, faster on CPU
        }
        
        with torch.no_grad():
            out = model_manager.blip_model.generate(**inputs, **generation_params)
        
        caption = model_manager.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.error(f"Error in image summarization: {str(e)}")
        return "Error processing image"


# --- TABLE EXTRACTION ---
def convert_df_to_rows(df: pd.DataFrame) -> list[str]:
    """Convert DataFrame to structured text rows"""
    if df.empty:
        return []
    
    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]
    
    rows = []
    for _, rec in df.iterrows():
        cells = ["" if pd.isna(v) else str(v).strip() for v in rec]
        pairs = [f"{hdr}: {val}" for hdr, val in zip(df.columns, cells) if val and val != ""]
        if pairs:
            rows.append(" | ".join(pairs))
    return rows


def extract_tables_text(pdf_path: str) -> list[str]:
    rows = []
    try:
        for tbl in camelot.read_pdf(pdf_path, pages="all", flavor="lattice"):
            df = tbl.df
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            rows.extend(convert_df_to_rows(df))
    except Exception:
        pass
    try:
        for tbl in camelot.read_pdf(pdf_path, pages="all", flavor="stream"):
            df = tbl.df
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            rows.extend(convert_df_to_rows(df))
    except Exception:
        pass
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                df = pd.DataFrame(table[1:], columns=table[0])
                rows.extend(convert_df_to_rows(df))
    return rows


# --- IMAGE EXTRACTION & SUMMARIZATION ---
def extract_image_summaries(pdf_path: str) -> list[str]:
    """Extract and summarize images from PDF with CPU optimizations"""
    if not model_manager.load_models():
        return []
    
    summaries = []
    try:
        doc = fitz.open(pdf_path)
        for pno, page in enumerate(doc, start=1):
            # Get page as image with lower DPI for CPU processing
            dpi = 100 if device == "cpu" else 150
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Convert color space
            if pix.n == 4:  # BGRA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Detect objects/images using YOLO with CPU optimizations
            yolo_params = {
                "verbose": False,
                "conf":  0.5,
                "device": device
            }
            
            results = model_manager.yolo(img, **yolo_params)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                # Limit number of images processed per page on CPU
                max_images = len(boxes)
                boxes = boxes[:max_images]
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    # Extract region and convert to PIL
                    region = img[y1:y2, x1:x2]
                    if region.size > 0:  # Check if region is valid
                        pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
                        summary = summarize_image(pil_image)
                        summaries.append(f"Page {pno} Image {i+1}: {summary}")
                        
                        # Add small delay for CPU processing
                        if device == "cpu" and i > 0:
                            import time
                            time.sleep(0.1)  # Small delay to prevent CPU overload
        
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting images from {pdf_path}: {str(e)}")
        if device == "cpu":
            logger.info("Consider disabling image processing for faster CPU performance")
    
    return summaries


# --- TEXT EXTRACTION ---
def extract_page_text(pdf_path: str) -> list[str]:
    """Extract text content from PDF pages"""
    texts = []
    try:
        doc = fitz.open(pdf_path)
        for pno, page in enumerate(doc, start=1):
            raw_text = page.get_text("text").strip()
            if raw_text:
                texts.append(f"Page {pno} Text: {raw_text}")
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
    
    return texts


# --- CHROMA CHECK & INGESTION ---
def is_populated(directory: str) -> bool:
    """Check if ChromaDB directory is populated"""
    if not os.path.exists(directory):
        return False
    
    # Check for ChromaDB files
    chroma_files = [
        f for f in os.listdir(directory) 
        if f.endswith(('.db', '.sqlite', '.sqlite3')) or 'chroma' in f.lower()
    ]
    return len(chroma_files) > 0

def process_partition(partition_name: str) -> bool:
    """Process a single partition and create its ChromaDB"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing partition: {partition_name}")
    logger.info(f"{'='*50}")
    
    # Set up paths for this partition
    partition_pdf_dir = os.path.join(base_pdfs_dir, partition_name)
    partition_persist_dir = os.path.join(base_persist_directory, partition_name)
    
    # Check if partition directory exists
    if not os.path.exists(partition_pdf_dir):
        logger.warning(f"Partition directory {partition_pdf_dir} does not exist. Skipping...")
        return False
    
    # Get all PDF paths for this partition
    pdf_paths = [
        os.path.join(partition_pdf_dir, f) 
        for f in os.listdir(partition_pdf_dir) 
        if f.lower().endswith(".pdf")
    ]
    
    if not pdf_paths:
        logger.warning(f"No PDFs found in {partition_pdf_dir}. Skipping...")
        return False
    
    total_pdfs = len(pdf_paths)
    logger.info(f"Found {total_pdfs} PDFs in {partition_name}")
    
    # Create persist directory for this partition
    os.makedirs(partition_persist_dir, exist_ok=True)
    
    # Check if already populated
    if is_populated(partition_persist_dir):
        logger.info(f"ChromaDB for {partition_name} already populated. Skipping...")
        return True
    
    # Process PDFs in this partition
    docs = []
    successful_pdfs = 0
    
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        logger.info(f"  Processing PDF {idx}/{total_pdfs}: {os.path.basename(pdf_path)}")
        
        try:
            # Extract different types of content
            page_texts = extract_page_text(pdf_path)
            table_texts = extract_tables_text(pdf_path)
            image_summaries = extract_image_summaries(pdf_path)
            
            # Combine all content
            all_content = page_texts + table_texts + image_summaries
            
            # Create documents from chunks
            for content in all_content:
                if content and len(content.strip()) > 20:  # Skip very short content
                    chunks = splitter.split_text(content)
                    for chunk in chunks:
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": os.path.basename(pdf_path),
                                    "partition": partition_name,
                                    "full_path": pdf_path,
                                    "content_type": "text" if "Text:" in content else 
                                                  "table" if "Table:" in content else "image"
                                }
                            )
                        )
            
            successful_pdfs += 1
            
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {successful_pdfs}/{total_pdfs} PDFs")
    
    if docs:
        try:
            # Create ChromaDB for this partition with explicit collection name
            logger.info(f"Creating ChromaDB with {len(docs)} chunks...")
            
            store = Chroma.from_documents(
                documents=docs,
                embedding=embedding_model,
                persist_directory=partition_persist_dir,
                collection_name=partition_name  # Explicit collection name
            )
            
            # Persist the database
            store.persist()
            logger.info(f"Successfully created ChromaDB for {partition_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating ChromaDB for {partition_name}: {str(e)}")
            return False
    else:
        logger.warning(f"No documents extracted for {partition_name}")
        return False


# --- MAIN PROCESSING LOOP ---
def main():
    logger.info("Starting multi-partition ChromaDB ingestion...")
    logger.info(f"Base PDF directory: {base_pdfs_dir}")
    logger.info(f"Base persist directory: {base_persist_directory}")
    logger.info(f"Partitions to process: {partitions}")
    
    # Validate base directories
    if not os.path.exists(base_pdfs_dir):
        logger.error(f"Base PDF directory {base_pdfs_dir} does not exist!")
        return
    
    total_partitions = len(partitions)
    successful_partitions = []
    failed_partitions = []
    
    try:
        for partition in partitions:
            try:
                if process_partition(partition):
                    successful_partitions.append(partition)
                else:
                    failed_partitions.append(partition)
            except Exception as e:
                logger.error(f"Critical error processing partition {partition}: {str(e)}")
                failed_partitions.append(partition)
                continue
    
    finally:
        # Cleanup models
        model_manager.cleanup()
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"PROCESSING SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total partitions: {total_partitions}")
    logger.info(f"Successfully processed: {len(successful_partitions)}")
    logger.info(f"Failed: {len(failed_partitions)}")
    
    if successful_partitions:
        logger.info(f"\nSuccessful partitions:")
        for partition in successful_partitions:
            logger.info(f"{partition}")
    
    if failed_partitions:
        logger.info(f"\nFailed partitions:")
        for partition in failed_partitions:
            logger.info(f"{partition}")
    
    # Show created databases
    logger.info(f"\nCreated ChromaDB directories:")
    for partition in partitions:
        partition_dir = os.path.join(base_persist_directory, partition)
        if os.path.exists(partition_dir) and is_populated(partition_dir):
            file_count = len([f for f in os.listdir(partition_dir) if not f.startswith('.')])
            logger.info(f"{partition_dir} ({file_count} files)")
        else:
            logger.info(f"{partition_dir} (not created or empty)")

    return len(successful_partitions) > 0

# Export the main function explicitly
__all__ = ['main']

if __name__ == "__main__":
    main()
