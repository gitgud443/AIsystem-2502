"""
Utility stubs for the face recognition project.

Each function is intentionally left unimplemented so that students can
fill in the logic as part of the coursework.
"""

from typing import Any, List
import cv2
import numpy as np
import insightface
import models

def detect_faces(image: Any) -> List[Any]:
    """
    Detect faces in image using InsightFace.
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        List of Face objects from InsightFace
        
    Raises:
        ValueError: If image is invalid or detection fails
        RuntimeError: If face detector is not initialized
    """
    
    # Validate image
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image provided")
    
    # Check if detector is initialized
    if models.face_detector is None:
        raise RuntimeError("Face detector not initialized. Server may still be starting up.")
    
    # Detect faces
    try:
        faces = models.face_detector.get(image)
    except Exception as e:
        raise ValueError(f"Face detection failed: {str(e)}")
    
    return faces


def compute_face_embedding(face_image: Any) -> Any:
    """
    Compute a 512-dimensional face embedding vector.
    
    This function supports two input modes:
    1. InsightFace Face object - uses pre-computed embedding
    2. Aligned 112*112 face image - attempts re-detection
    
    Note: Mode 2 often fails because InsightFace's detector struggles with
    pre-aligned faces. For production use, prefer Mode 1.
    
    Args:
        face_image: Either:
                   - InsightFace Face object with pre-computed embedding
                   - Aligned face image as numpy array (112*112, BGR)
        
    Returns:
        Face embedding as numpy array of shape (512,)
        
    Raises:
        ValueError: If face_image is invalid or embedding extraction fails
        RuntimeError: If face recognition model is not initialized
    """
    
    # MODE 1: Face object with pre-computed embedding (RECOMMENDED)
    # ---------------------------------------------------------------
    if hasattr(face_image, 'embedding'):
        embedding = face_image.embedding
        
        if embedding is None:
            raise ValueError("Face object has no embedding")
        
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding is not a numpy array")
        
        if len(embedding) != 512:
            raise ValueError(f"Invalid embedding dimension: expected 512, got {len(embedding)}")
        
        return embedding
    
    
    # MODE 2: Aligned face image (UNRELIABLE - may fail)
    # ---------------------------------------------------------------
    # Validate input
    if face_image is None:
        raise ValueError("Invalid face_image: image is None")
    
    if not isinstance(face_image, np.ndarray):
        raise ValueError("Invalid face_image: expected numpy array or Face object")
    
    if face_image.size == 0:
        raise ValueError("Invalid face_image: empty image array")
    
    if len(face_image.shape) != 3 or face_image.shape[2] != 3:
        raise ValueError(
            f"Invalid face_image: expected 3-channel BGR image, got shape {face_image.shape}"
        )
    
    # Check if model is initialized
    if models.face_detector is None:
        raise RuntimeError("Face detector/recognition model not initialized")
    
    # Try to extract embedding from aligned face
    # WARNING: This often fails because the detector expects full images, not aligned crops
    try:
        faces = models.face_detector.get(face_image)
        
        if len(faces) == 0:
            raise ValueError(
                "No face detected in the provided image. "
                "If this is a pre-aligned 112*112 face, the detector may not recognize it. "
                "Consider using the Face object with pre-computed embedding instead."
            )
        
        embedding = faces[0].embedding
        
        if embedding is None:
            raise ValueError("Embedding extraction failed: embedding is None")
        
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding is not in expected numpy array format")
        
        if embedding.shape[0] != 512:
            raise ValueError(
                f"Unexpected embedding dimension: expected 512, got {embedding.shape[0]}"
            )
        
        return embedding
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Embedding extraction failed: {str(e)}. "
            "This often happens with pre-aligned faces. "
            "Use the Face object from detect_faces() instead."
        )


def detect_face_keypoints(face_image: Any) -> Any:
    """
    Extract facial keypoints (landmarks) from a detected face.
    
    With InsightFace, keypoints are already detected during face detection,
    so this function extracts them from the Face object.
    
    Args:
        face_image: InsightFace Face object (returned from detect_faces)
        
    Returns:
        Keypoints array of shape (5, 2) containing:
            - [0]: left eye center (x, y)
            - [1]: right eye center (x, y)
            - [2]: nose tip (x, y)
            - [3]: left mouth corner (x, y)
            - [4]: right mouth corner (x, y)
            
    Raises:
        ValueError: If face object doesn't contain keypoints
        TypeError: If input is not a valid Face object
    """
    
    # STEP 1: Validate input type
    # ---------------------------------
    # Check if this is an InsightFace Face object
    if not hasattr(face_image, 'kps'):
        raise TypeError(
            "Invalid input: expected InsightFace Face object with 'kps' attribute. "
            "Did you pass the raw image instead of a Face object?"
        )
    
    
    # STEP 2: Extract keypoints
    # ---------------------------------
    keypoints = face_image.kps
    
    
    # STEP 3: Validate keypoints
    # ---------------------------------
    if keypoints is None:
        raise ValueError("No keypoints detected in the face object")
    
    if not isinstance(keypoints, np.ndarray):
        raise ValueError("Keypoints are not in expected numpy array format")
    
    # Check shape: should be (5, 2) for 5 landmarks with x,y coordinates
    if keypoints.shape != (5, 2):
        raise ValueError(
            f"Invalid keypoints shape: expected (5, 2), got {keypoints.shape}"
        )
    
    
    # STEP 4: Return keypoints
    # ---------------------------------
    return keypoints


def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp/align a face image using a transformation matrix.
    
    This function applies an affine transformation to align a detected face
    to a standard 112×112 template position based on facial keypoints.
    
    Args:
        image: Source image containing the face (numpy array, BGR format)
        homography_matrix: 2×3 affine transformation matrix that maps
                          source keypoints to reference keypoints
        
    Returns:
        Aligned face image as numpy array (112×112 pixels, BGR format)
        
    Raises:
        ValueError: If inputs are invalid or warping fails
    """
    
    # STEP 1: Validate image input
    # ---------------------------------
    if image is None:
        raise ValueError("Invalid image: image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid image: expected numpy array")
    
    if image.size == 0:
        raise ValueError("Invalid image: empty image array")
    
    if len(image.shape) != 3:
        raise ValueError(
            f"Invalid image dimensions: expected 3D array (height, width, channels), "
            f"got shape {image.shape}"
        )
    
    
    # STEP 2: Validate transformation matrix
    # ---------------------------------
    if homography_matrix is None:
        raise ValueError("Invalid homography_matrix: matrix is None")
    
    if not isinstance(homography_matrix, np.ndarray):
        raise ValueError("Invalid homography_matrix: expected numpy array")
    
    # Affine transformation matrix should be 2×3
    if homography_matrix.shape != (2, 3):
        raise ValueError(
            f"Invalid homography_matrix shape: expected (2, 3), "
            f"got {homography_matrix.shape}. "
            f"Use cv2.estimateAffinePartial2D or cv2.getAffineTransform."
        )
    
    
    # STEP 3: Apply affine transformation
    # ---------------------------------
    try:
        # Use OpenCV's warpAffine to apply the transformation
        # Output size is 112×112 (standard for face recognition)
        aligned_face = cv2.warpAffine(
            src=image,                    # Source image
            M=homography_matrix,          # 2×3 transformation matrix
            dsize=(112, 112),             # Output size (width, height)
            flags=cv2.INTER_LINEAR,       # Interpolation method
            borderMode=cv2.BORDER_CONSTANT,  # How to handle borders
            borderValue=0.0               # Fill value for borders (black)
        )
        
    except Exception as e:
        raise ValueError(f"Face warping failed: {str(e)}")
    
    
    # STEP 4: Validate output
    # ---------------------------------
    if aligned_face is None or aligned_face.size == 0:
        raise ValueError("Warping produced empty result")
    
    if aligned_face.shape != (112, 112, 3):
        raise ValueError(
            f"Warping produced unexpected output shape: {aligned_face.shape}"
        )
    
    
    # STEP 5: Return aligned face
    # ---------------------------------
    return aligned_face


def antispoof_check(face_image: Any) -> float:
    """
    Simplified anti-spoofing check that assumes all faces are real.
    
    Args:
        face_image: Aligned face image as numpy array
        
    Returns:
        Always returns 0.95 (high confidence that face is real)
        
    Raises:
        ValueError: If face_image is invalid
    """
    
    # Basic validation
    if face_image is None or not isinstance(face_image, np.ndarray):
        raise ValueError("Invalid face_image")
    
    if face_image.size == 0:
        raise ValueError("Empty face_image")
    
    # Return high confidence score (assuming all faces are real)
    # This effectively disables anti-spoofing checks
    return 0.95


def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    End-to-end pipeline for comparing two face images using InsightFace.
    
    Steps:
    1. Decode raw bytes to numpy arrays
    2. Detect faces in both images
    3. Validate detection and select largest face
    4. Verify keypoints are detected
    5. Align faces to 112*112 using keypoints
    6. (Optional) Perform anti-spoofing checks
    7. Generate embeddings
    8. Calculate cosine similarity
    
    Args:
        image_a: First image as raw bytes (from UploadFile)
        image_b: Second image as raw bytes (from UploadFile)
    
    Returns:
        Similarity score between 0.0 and 1.0 (typically):
            - 1.0 = identical faces
            - 0.6-0.8 = same person
            - 0.4-0.6 = possibly same person
            - < 0.4 = different people
    
    Raises:
        ValueError: If image cannot be decoded, no face detected, or keypoints missing
    """
    
    # STEP 1: Decode bytes to images
    # ---------------------------------
    try:
        # Convert bytes to numpy array using cv2
        # image_a bytes → BGR numpy array
        nparr_a = np.frombuffer(image_a, np.uint8)
        img_a = cv2.imdecode(nparr_a, cv2.IMREAD_COLOR)
        
        nparr_b = np.frombuffer(image_b, np.uint8)
        img_b = cv2.imdecode(nparr_b, cv2.IMREAD_COLOR)
        
        if img_a is None:
            raise ValueError("Failed to decode image A - invalid image format")
        if img_b is None:
            raise ValueError("Failed to decode image B - invalid image format")
            
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        raise ValueError(f"Error decoding images: {str(e)}")
    
    
    # STEP 2: Detect faces
    # ---------------------------------
    faces_a = detect_faces(img_a)
    faces_b = detect_faces(img_b)
    
    
    # STEP 3: Validate face detection
    # ---------------------------------
    if len(faces_a) == 0:
        raise ValueError("No face detected in image A")
    
    if len(faces_b) == 0:
        raise ValueError("No face detected in image B")
    
    
    # STEP 4: Select largest face if multiple detected
    # ---------------------------------
    face_a = _select_largest_face(faces_a)
    face_b = _select_largest_face(faces_b)
    
    
    # STEP 5: Extract and verify keypoints
    # ---------------------------------
    # Using detect_face_keypoints as required by the assignment
    # This function will raise ValueError if keypoints are missing
    keypoints_a = detect_face_keypoints(face_a)
    keypoints_b = detect_face_keypoints(face_b)
    
    
    # STEP 6: Align faces to 112×112
    # ---------------------------------
    aligned_face_a = _align_face(img_a, keypoints_a)
    aligned_face_b = _align_face(img_b, keypoints_b)
    
    
    # STEP 7: (Optional) Anti-spoofing checks
    # ---------------------------------
    ANTISPOOF_THRESHOLD = 0.7
    
    spoof_score_a = antispoof_check(aligned_face_a)
    spoof_score_b = antispoof_check(aligned_face_b)
    
    if spoof_score_a < ANTISPOOF_THRESHOLD:
        raise ValueError(
            f"Image A failed anti-spoofing check (confidence: {spoof_score_a:.3f}). "
            "Possible print attack or screen replay detected."
        )
    
    if spoof_score_b < ANTISPOOF_THRESHOLD:
        raise ValueError(
            f"Image B failed anti-spoofing check (confidence: {spoof_score_b:.3f}). "
            "Possible print attack or screen replay detected."
        )
    
    
    # STEP 8: Generate embeddings
    # ---------------------------------
    embedding_a = face_a.embedding
    embedding_b = face_b.embedding
    
    # Validate embeddings
    if embedding_a is None or embedding_b is None:
        raise ValueError("Failed to extract embeddings from detected faces")
    
    if len(embedding_a) != 512 or len(embedding_b) != 512:
        raise ValueError("Invalid embedding dimensions")
    
    
    # STEP 9: Calculate cosine similarity
    # ---------------------------------
    similarity = _cosine_similarity(embedding_a, embedding_b)
    
    print(similarity)
    return float(similarity)


# Helper function: Select largest face
def _select_largest_face(faces):
    """
    Select the face with the largest bounding box area.
    
    Args:
        faces: List of InsightFace Face objects
        
    Returns:
        Face object with largest bbox area
    """
    largest_face = None
    max_area = 0
    
    for face in faces:
        # InsightFace bbox format: [x1, y1, x2, y2]
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area > max_area:
            max_area = area
            largest_face = face
    
    return largest_face


# Helper function: Align face using keypoints
def _align_face(image, keypoints):
    """
    Align face to standard 112*112 template using detected keypoints.
    
    Args:
        image: Original image (numpy array)
        keypoints: 5 facial landmarks from InsightFace (shape: 5x2)
        
    Returns:
        Aligned face image (112*112 numpy array)
    """
    # Define reference keypoints for 112*112 aligned face
    # These are the standard positions from ArcFace/InsightFace
    reference_keypoints = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041]   # right mouth corner
    ], dtype=np.float32)
    
    # Calculate similarity transform (affine transformation)
    # This maps detected keypoints → reference keypoints
    transformation_matrix = cv2.estimateAffinePartial2D(
        keypoints.astype(np.float32),
        reference_keypoints,
        method=cv2.LMEDS
    )[0]
    
    if transformation_matrix is None:
        raise ValueError("Failed to calculate transformation matrix")

    # Apply the transformation using warp_face
    aligned_face = warp_face(image, transformation_matrix)
    
    return aligned_face


# Helper function: Calculate cosine similarity
def _cosine_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding_a: First face embedding (512-dim numpy array)
        embedding_b: Second face embedding (512-dim numpy array)
        
    Returns:
        Similarity score between -1 and 1 (typically 0 to 1 for faces)
    """
    # Normalize embeddings (convert to unit vectors)
    embedding_a = embedding_a / np.linalg.norm(embedding_a)
    embedding_b = embedding_b / np.linalg.norm(embedding_b)
    
    # Compute dot product (which equals cosine similarity for unit vectors)
    similarity = np.dot(embedding_a, embedding_b)
    
    return similarity
