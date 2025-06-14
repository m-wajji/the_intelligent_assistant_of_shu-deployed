from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import shutil, os
import traceback
import logging

from app.main import process_agent_query, process_audio_file
try:
    from app.data_ingestion import main as run_data_ingestion
    DATA_INGESTION_AVAILABLE = True
except ImportError:
    DATA_INGESTION_AVAILABLE = False
    logging.warning("data_ingestion.py not found. Data ingestion will be skipped.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
UPLOAD_DIR = "uploads"
AUDIO_RESPONSE_DIR = "audio_responses"
PERSIST_DIR = "shu_partitioned_db/chroma_dbs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_RESPONSE_DIR, exist_ok=True)

# Check if persistent directory exists and run data ingestion if needed
def check_and_run_data_ingestion():
    """Check if persistent data exists, if not run data ingestion"""
    try:
        if not DATA_INGESTION_AVAILABLE:
            logger.info("Data ingestion not available. Skipping check.")
            return
            
        if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
            logger.info("Persistent directory not found or empty. Running data ingestion...")
            run_data_ingestion()
            logger.info("Data ingestion completed.")
        else:
            logger.info("Persistent directory exists and contains data. Skipping data ingestion.")
    except Exception as e:
        logger.error(f"Error during data ingestion check: {str(e)}")
        # Don't raise the exception to prevent app startup failure
        logger.warning("Continuing without data ingestion...")

# Lifespan event handler for startup and shutdown tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    check_and_run_data_ingestion()
    yield
    # Shutdown (if you need cleanup tasks)
    logger.info("Shutting down FastAPI application...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Session tracking for text bot
MAX_QUERIES_PER_SESSION = 3
session_queries = defaultdict(int)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        session_id = request.headers.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")

        if session_queries[session_id] >= MAX_QUERIES_PER_SESSION:
            raise HTTPException(status_code=403, detail="Query limit exceeded.")

        answer = process_agent_query(chat_request.question)
        session_queries[session_id] += 1
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...), request: Request = None):
    try:
        session_id = request.headers.get("session_id") if request else None
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")

        if session_queries[session_id] >= MAX_QUERIES_PER_SESSION:
            raise HTTPException(status_code=403, detail="Query limit exceeded.")

        # Validate file type
        if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.webm')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"Saving audio file to: {file_path}")
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved successfully, size: {os.path.getsize(file_path)} bytes")

        # Process the audio file
        logger.info("Starting audio processing...")
        result = process_audio_file(file.filename)
        
        if "error" in result:
            logger.error(f"Audio processing error: {result['error']}")
            return JSONResponse(content=result, status_code=400)

        # Increment after successful processing
        session_queries[session_id] += 1
        logger.info("Audio processing completed successfully")
        
        return {
            "transcription": result["transcription"],
            "translated_text": result.get("translated_text"),
            "answer": result["answer"],
            "pdf_url": result.get("pdf_url"),
            "audio_response_url": result["audio_response_url"],
        }
        
    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Upload audio error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": f"Server error: {str(e)}"}, status_code=500
        )
    finally:
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                logger.info(f"Cleaning up uploaded file: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error: {cleanup_error}")

@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    try:
        audio_file = os.path.join(AUDIO_RESPONSE_DIR, filename)
        if os.path.exists(audio_file):
            return FileResponse(audio_file, media_type="audio/mpeg", filename=filename)
        return JSONResponse(content={"error": "Audio file not found!"}, status_code=404)
    except Exception as e:
        logger.error(f"Download audio error: {str(e)}")
        return JSONResponse(content={"error": f"Download error: {str(e)}"}, status_code=500)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_ingestion_available": DATA_INGESTION_AVAILABLE,
        "persist_dir_exists": os.path.exists(PERSIST_DIR),
        "persist_dir_populated": os.path.exists(PERSIST_DIR) and bool(os.listdir(PERSIST_DIR)) if os.path.exists(PERSIST_DIR) else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)