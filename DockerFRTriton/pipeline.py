from typing import Any, Tuple, List, Dict
import numpy as np
from io import BytesIO
from PIL import Image
import math

from triton_service import (
    run_detector_inference,
    run_recognition_inference,
    preprocess_for_detector,
    FR_MODEL_IMAGE_SIZE
)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = np.linalg.norm(vec_a)
    b_norm = np.linalg.norm(vec_b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (a_norm * b_norm))


def postprocess_scrfd(raw_outputs: Dict[str, np.ndarray], img_shape: Tuple[int, int], conf_threshold: float = 0.4, nms_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    Post-process SCRFD raw outputs (9 tensors: score, bbox, kps for each of 3 strides).
    Outputs are flattened, no batch dim visible after squeeze.
    """
    # Sort keys to get consistent order
    output_names = sorted(raw_outputs.keys())
    
    # Expected grouping: score, bbox, kps for stride32, stride16, stride8 (or reverse — we'll match by shape)
    scores = []
    bboxes = []
    # landmarks = []  # optional, ignore for now
    
    for key in output_names:
        tensor = raw_outputs[key]
        shape = tensor.shape
        if len(shape) == 2 and shape[1] == 1:  # scores
            scores.append(tensor.squeeze(1))  # (N,)
        elif len(shape) == 2 and shape[1] == 4:  # bboxes
            bboxes.append(tensor)  # (N, 4)
        elif len(shape) == 2 and shape[1] == 10:  # landmarks
            pass  # ignore
    
    # Should have 3 scores and 3 bboxes
    assert len(scores) == 3 and len(bboxes) == 3, f"Unexpected number of score/bbox tensors: {len(scores)}/{len(bboxes)}"
    
    h, w = img_shape
    detections = []
    
    for score, bbox in zip(scores, bboxes):
        # Sigmoid for confidence (SCRFD uses sigmoid)
        score = 1 / (1 + np.exp(-score))  # (N,)
        
        # Filter threshold
        keep = score > conf_threshold
        score = score[keep]
        bbox = bbox[keep]
        
        if len(score) == 0:
            continue
        
        # BBoxes are usually in absolute pixel coordinates already (common in InsightFace ONNX)
        # If boxes look tiny/off, uncomment scaling:
        # bbox[:, [0, 2]] *= w
        # bbox[:, [1, 3]] *= h
        
        # Clamp
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, w)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, h)
        
        for i in range(len(score)):
            x1, y1, x2, y2 = bbox[i]
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'score': float(score[i])
            })
    
    # Sort by score descending
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    print(f"[Detector] Found {len(detections)} raw detections (threshold={conf_threshold})")
    
    # Optional: simple NMS (greedy by score)
    if detections:
        # Implement basic NMS or skip for single-face images
        pass  # for assignment, single best face is fine
    
    return detections


def detect_and_crop_face(image_bytes: bytes, client: Any) -> np.ndarray:
    """
    Detect face in image using Triton detector and return cropped face.
    
    Args:
        image_bytes: Raw image bytes
        client: Triton client instance
    
    Returns:
        Cropped and aligned face as numpy array (H, W, 3) in RGB
    
    Raises:
        ValueError: If no face is detected
    """
    # Run detection on Triton
    preprocessed = preprocess_for_detector(image_bytes)
    raw_outputs = run_detector_inference(client, preprocessed)
    
    # Get original image size
    with Image.open(BytesIO(image_bytes)) as img:
        original_size = img.size  # (w, h)
        img_shape = img.height, img.width  # (h, w)
    
    detections = postprocess_scrfd(raw_outputs, img_shape)
    
    if len(detections) == 0:
        raise ValueError("No face detected in the image")
    
    best_detection = max(detections, key=lambda x: x['score'])
    
    # Load image and crop face
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        
        # Get bounding box
        x1, y1, x2, y2 = best_detection['bbox']
        
        # Add some padding to the crop (optional, helps with alignment)
        padding = 0.1  # 10% padding
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, int(x1 - width * padding))
        y1 = max(0, int(y1 - height * padding))
        x2 = min(img.width, int(x2 + width * padding))
        y2 = min(img.height, int(y2 + height * padding))
        
        # Crop face
        face_crop = img.crop((x1, y1, x2, y2))
        
        # Resize to model input size
        face_crop = face_crop.resize(FR_MODEL_IMAGE_SIZE)

    face_crop = face_crop.resize(FR_MODEL_IMAGE_SIZE)

    # Convert to float32 and apply standard ArcFace normalization
    face_np = np.asarray(face_crop, dtype=np.float32)
    face_np = (face_np - 127.5) / 128.0   # Critical: mean 127.5, std 128.0

    # Transpose to CHW and add batch dim
    face_np = np.transpose(face_np, (2, 0, 1))  # HWC -> CHW
    face_np = np.expand_dims(face_np, axis=0)   # (1, 3, 112, 112)
    
    return face_np


def get_face_embedding(image_bytes: bytes, client: Any) -> np.ndarray:
    """
    Complete pipeline: detect face, crop, and extract embedding using Triton.
    
    Args:
        image_bytes: Raw image bytes
        client: Triton client instance
    
    Returns:
        Face embedding vector (512,)
    
    Raises:
        ValueError: If no face is detected or embedding extraction fails
    """
    # Step 1: Detect and crop face
    face_crop = detect_and_crop_face(image_bytes, client)
    
    # Step 2: Extract embedding using Triton FR model
    embedding = run_recognition_inference(client, face_crop)
    
    return embedding


def get_embeddings(client: Any, image_a: bytes, image_b: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for two images using the full Triton pipeline.
    
    This function:
    1. Detects faces in both images using Triton detector
    2. Crops and aligns the detected faces
    3. Extracts embeddings using Triton FR model
    
    All model inference happens on Triton server.
    
    Args:
        client: Triton client instance
        image_a: First image as bytes
        image_b: Second image as bytes
    
    Returns:
        Tuple of (embedding_a, embedding_b) as numpy arrays
    
    Raises:
        ValueError: If face detection or embedding extraction fails
    """
    emb_a = get_face_embedding(image_a, client)
    emb_b = get_face_embedding(image_b, client)
    
    return emb_a, emb_b


def calculate_face_similarity(client: Any, image_a: bytes, image_b: bytes) -> float:
    """
    End-to-end face similarity calculation using Triton models.
    
    Pipeline:
    1. Detect faces in both images (Triton detector)
    2. Crop and align faces
    3. Extract embeddings (Triton FR model)
    4. Calculate cosine similarity
    
    Args:
        client: Triton client instance
        image_a: First image as bytes
        image_b: Second image as bytes
    
    Returns:
        Cosine similarity score between 0 and 1
    
    Raises:
        ValueError: If face detection fails or images are invalid
    """
    # Get embeddings for both images
    emb_a, emb_b = get_embeddings(client, image_a, image_b)
    
    # Calculate similarity
    similarity = _cosine_similarity(emb_a, emb_b)
    
    return similarity


def detect_all_faces(image_bytes: bytes, client: Any) -> List[Dict[str, Any]]:
    """
    Detect all faces in an image using Triton detector.
    
    Args:
        image_bytes: Raw image bytes
        client: Triton client instance
    
    Returns:
        List of detections, each containing:
        - bbox: [x1, y1, x2, y2]
        - score: confidence score
        - landmarks: facial landmarks (10 values for 5 points)
    """
    preprocessed = preprocess_for_detector(image_bytes)
    detections = run_detector_inference(client, preprocessed)
    return detections


def extract_embeddings_for_all_faces(image_bytes: bytes, client: Any) -> List[np.ndarray]:
    """
    Detect all faces in an image and extract embeddings for each.
    
    Args:
        image_bytes: Raw image bytes
        client: Triton client instance
    
    Returns:
        List of embedding vectors (512,) for each detected face
    
    Raises:
        ValueError: If no faces are detected
    """
    # Detect all faces
    preprocessed = preprocess_for_detector(image_bytes)
    detections = run_detector_inference(client, preprocessed)
    
    if len(detections) == 0:
        raise ValueError("No faces detected in the image")
    
    embeddings = []
    
    # Load image once
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        
        # Extract embedding for each detected face
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop face
            face_crop = img.crop((x1, y1, x2, y2))
            face_crop = face_crop.resize(FR_MODEL_IMAGE_SIZE)

            # Convert to float32 and apply standard ArcFace normalization
            face_np = np.asarray(face_crop, dtype=np.float32)
            face_np = (face_np - 127.5) / 128.0   # Critical: mean 127.5, std 128.0

            # Transpose to CHW and add batch dim
            face_np = np.transpose(face_np, (2, 0, 1))  # HWC -> CHW
            face_np = np.expand_dims(face_np, axis=0)   # (1, 3, 112, 112)
            
            # Get embedding
            embedding = run_recognition_inference(client, face_np)
            embeddings.append(embedding)
    
    return embeddings
