# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional # Added Optional
import logging
import time
import os # For environment variables if needed

# --- Basic Logging Configuration ---
# This will make logs from libraries like huggingface_hub more visible
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "Qodo/Qodo-Embed-1-1.5B")
DEVICE = os.getenv("DEVICE", "cpu") # Can be 'cuda' if GPU is available

app = FastAPI()

model_instance: Optional[SentenceTransformer] = None

logger.info(f"FastAPI application startup: Attempting to load model '{MODEL_NAME}' on device '{DEVICE}'.")
logger.info("This may take a while if the model needs to be downloaded...")
start_time = time.time()
try:
    # trust_remote_code is needed for Qodo models
    model_instance = SentenceTransformer(MODEL_NAME, trust_remote_code=("Qodo" in MODEL_NAME), device=DEVICE)
    end_time = time.time()
    logger.info(f"Model '{MODEL_NAME}' loaded successfully on device '{DEVICE}' in {end_time - start_time:.2f} seconds.")
except Exception as e:
    logger.error(f"Fatal error loading model '{MODEL_NAME}': {e}", exc_info=True)
    # The application might not be usable if the model fails to load.
    # Consider how to handle this (e.g., exit, or run in a degraded state).
    model_instance = None # Ensure it's None if loading failed


# --- Request and Response Models ---
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class UsageData(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageData


# --- API Endpoint ---
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    global model_instance
    if model_instance is None:
        logger.error("Model is not loaded or failed to load. Cannot process embedding request.")
        # from fastapi import HTTPException
        # raise HTTPException(status_code=503, detail="Model not available")
        # For now, returning an empty response or a specific error structure:
        requested_model_name = request.model if request.model else MODEL_NAME
        return EmbeddingResponse(data=[], model=requested_model_name, usage=UsageData(prompt_tokens=0, total_tokens=0))

    input_texts = [request.input] if isinstance(request.input, str) else request.input
    
    # Generate embeddings
    embeddings = model_instance.encode(input_texts)

    response_data: List[EmbeddingData] = []
    for i, emb_vector in enumerate(embeddings):
        response_data.append(EmbeddingData(embedding=emb_vector.tolist(), index=i))

    # Basic token counting (very approximate)
    prompt_tokens = sum(len(text.split()) for text in input_texts)
    total_tokens = prompt_tokens
    
    current_model_name = request.model if request.model else MODEL_NAME

    return EmbeddingResponse(
        data=response_data,
        model=current_model_name,
        usage=UsageData(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
    )

# --- Health Check (Updated) ---
@app.get("/health")
async def health():
    global model_instance
    model_status = "loaded" if model_instance is not None else "not_loaded_or_failed"
    if model_instance is None:
        logger.warning(f"Health check: Model '{MODEL_NAME}' is {model_status}.")
    return {"status": "ok", "model_name": MODEL_NAME, "model_status": model_status, "device": DEVICE}