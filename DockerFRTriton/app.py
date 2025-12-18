from fastapi import FastAPI, File, HTTPException, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any

from triton_service import (
    prepare_model_repository,
    create_triton_client,
)
from pipeline import (
    calculate_face_similarity,
    get_face_embedding,
    detect_all_faces,
    extract_embeddings_for_all_faces,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prepare model repository (generates config.pbtxt files for both models)
    model_repo = Path("model_repository")
    prepare_model_repository(model_repo)
    print("[startup] Model repository prepared")

    # Create Triton client and wait for server to be ready
    global triton_client
    triton_client = create_triton_client(url="triton:8000")  # Docker service name

    import time
    max_attempts = 30
    for i in range(max_attempts):
        if triton_client.is_server_live():
            print("[startup] Connected to Triton server!")
            break
        print(f"[startup] Waiting for Triton... ({i+1}/{max_attempts})")
        time.sleep(2)
    else:
        raise RuntimeError("Failed to connect to Triton server")

    yield  # Application runs here

    # Optional cleanup
    if triton_client:
        triton_client.close()

app = FastAPI(lifespan=lifespan)


app = FastAPI(
    title="Face Recognition with Triton",
    description="Face recognition system using Triton Inference Server. All models (detector + FR) run on Triton.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    shape: List[int]


class SimilarityResponse(BaseModel):
    similarity: float


class DetectionResponse(BaseModel):
    num_faces: int
    detections: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    triton_server: str
    triton_client: str


@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint."""
    return {
        "message": "Face Recognition API with Triton Inference Server",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint."""
    server_status = "running" if triton_server is not None else "not started"
    client_status = "connected" if triton_client is not None else "not connected"
    
    overall_status = "ok" if (triton_server is not None and triton_client is not None) else "degraded"
    
    return {
        "status": overall_status,
        "triton_server": server_status,
        "triton_client": client_status
    }


@app.post("/embedding", response_model=EmbeddingResponse, tags=["Face Recognition"])
async def get_embedding(
    image: UploadFile = File(..., description="Face image file"),
):
    """
    Extract face embedding from an image.
    
    Pipeline:
    1. Detect face using Triton detector
    2. Crop and align face
    3. Extract 512-dim embedding using Triton FR model
    
    Returns:
        - embedding: 512-dimensional face embedding vector
        - shape: Shape of the embedding array
    """
    if triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="Triton server not ready. Please wait a moment and try again."
        )
    
    try:
        content = await image.read()
        
        # Get embedding using full pipeline (all on Triton)
        embedding = get_face_embedding(content, triton_client)
        
        return {
            "embedding": embedding.tolist(),
            "shape": list(embedding.shape)
        }
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Unexpected error in /embedding: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@app.post("/face-similarity", response_model=SimilarityResponse, tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(..., description="First face image"),
    image_b: UploadFile = File(..., description="Second face image"),
):
    """
    Compare two face images and return similarity score.
    
    Pipeline:
    1. Detect faces in both images (Triton detector)
    2. Crop and align faces
    3. Extract embeddings (Triton FR model)
    4. Calculate cosine similarity
    
    Returns:
        - similarity: Cosine similarity score (0 to 1, higher = more similar)
    """
    if triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="Triton server not ready. Please wait a moment and try again."
        )
    
    try:
        content_a = await image_a.read()
        content_b = await image_b.read()
        
        print(f"Processing images - A: {len(content_a)} bytes, B: {len(content_b)} bytes")
        
        # Calculate similarity using full Triton pipeline
        similarity = calculate_face_similarity(triton_client, content_a, content_b)
        
        print(f"Similarity calculated: {similarity:.4f}")
        
        return {"similarity": similarity}
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Unexpected error in /face-similarity: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@app.post("/detect-faces", response_model=DetectionResponse, tags=["Face Detection"])
async def detect_faces_endpoint(
    image: UploadFile = File(..., description="Image file to detect faces in"),
):
    """
    Detect all faces in an image using Triton detector.
    
    Returns:
        - num_faces: Number of faces detected
        - detections: List of detections with bbox, score, and landmarks
    """
    if triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="Triton server not ready. Please wait a moment and try again."
        )
    
    try:
        content = await image.read()
        
        # Detect faces using Triton
        detections = detect_all_faces(content, triton_client)
        
        return {
            "num_faces": len(detections),
            "detections": detections
        }
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Unexpected error in /detect-faces: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@app.post("/extract-all-embeddings", tags=["Face Recognition"])
async def extract_all_embeddings(
    image: UploadFile = File(..., description="Image file with one or more faces"),
):
    """
    Detect all faces in an image and extract embeddings for each.
    
    Pipeline:
    1. Detect all faces using Triton detector
    2. For each face: crop, align, and extract embedding (Triton FR model)
    
    Returns:
        - num_faces: Number of faces detected
        - embeddings: List of 512-dim embeddings for each face
    """
    if triton_client is None:
        raise HTTPException(
            status_code=503,
            detail="Triton server not ready. Please wait a moment and try again."
        )
    
    try:
        content = await image.read()
        
        # Extract embeddings for all faces
        embeddings = extract_embeddings_for_all_faces(content, triton_client)
        
        return {
            "num_faces": len(embeddings),
            "embeddings": [emb.tolist() for emb in embeddings]
        }
        
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Unexpected error in /extract-all-embeddings: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")