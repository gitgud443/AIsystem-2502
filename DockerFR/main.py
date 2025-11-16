from fastapi import FastAPI, File, HTTPException, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
from util import calculate_face_similarity
import insightface
import models

# Global variable to store the face detector
face_detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    
    # Startup
    print("="*60)
    print("APPLICATION STARTUP")
    print("="*60)
    print("Initializing face detector...")
    
    try:
        # Initialize detector
        models.face_detector = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        models.face_detector.prepare(ctx_id=0, det_size=(640, 640))
        
        # Verify it's actually set
        if models.face_detector is None:
            raise RuntimeError("Face detector is still None after initialization")
        
        print("✓ Face detector initialized successfully!")
        print(f"✓ Detector type: {type(models.face_detector)}")
        print("="*60)
        
    except Exception as e:
        print(f"✗ FATAL ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield  # Application runs here
    
    # Shutdown
    print("="*60)
    print("APPLICATION SHUTDOWN")
    print("="*60)
    models.face_detector = None

app = FastAPI(
    title="My FastAPI Service",
    description="A simple demo API running in Docker. Swagger is at /docs and ReDoc at /redoc.",
    version="0.1.0",
    docs_url="/docs",          # Swagger UI URL
    redoc_url="/redoc",        # ReDoc URL
    openapi_url="/openapi.json", # OpenAPI JSON spec URL
    lifespan=lifespan           # added lifespan for face detection initializing
)

class Echo(BaseModel):
    text: str


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello, FastAPI in Docker!"}


@app.get("/torch-version", tags=["info"])
def torch_version():
    return {"torch_version": torch.__version__}


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint."""
    detector_status = "initialized" if models.face_detector is not None else "not initialized"
    return {
        "status": "ok",
        "face_detector": detector_status
    }


@app.post("/echo", tags=["Demo"])
def echo(body: Echo):
    return {"you_sent": body.text}


@app.post("/face-similarity", tags=["Face Recognition"])
async def face_similarity(
    image_a: UploadFile = File(..., description="First face image file"),
    image_b: UploadFile = File(..., description="Second face image file"),
):
    """
    Compare two face images and return a similarity score between them.
    """
    # Check if detector is initialized
    if models.face_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Face detector not initialized. Server is still starting up. Please wait a moment and try again."
        )
    
    try:
        content_a = await image_a.read()
        content_b = await image_b.read()
        
        print(f"Processing images - A: {len(content_a)} bytes, B: {len(content_b)} bytes")
        
        similarity = calculate_face_similarity(content_a, content_b)
        
        print(f"Similarity calculated: {similarity:.4f}")
        
        return {"similarity": similarity}
        
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")
