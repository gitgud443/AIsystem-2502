"""
Test script for face recognition utilities.
"""
import cv2
import numpy as np
from util import (
    detect_faces,
    detect_face_keypoints,
    warp_face,
    compute_face_embedding,
    antispoof_check,
    calculate_face_similarity
)

def test_individual_functions():
    """Test each function step by step."""
    
    print("=" * 60)
    print("TESTING INDIVIDUAL FUNCTIONS")
    print("=" * 60)
    
    # Load a test image
    image_path = "media/face1.jpg"  # Change to your image path
    print(f"\n1. Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        print("Make sure the path is correct and the file exists.")
        return
    
    print(f"   Image loaded successfully - Shape: {image.shape}")
    
    
    # Test face detection
    print("\n2. Testing detect_faces()...")
    try:
        faces = detect_faces(image)
        print(f"   Detected {len(faces)} face(s)")
        
        if len(faces) == 0:
            print("   ERROR: No faces detected. Try a different image.")
            return
        
        for i, face in enumerate(faces):
            bbox = face.bbox
            print(f"   Face {i+1}: bbox={bbox}, confidence={face.det_score:.3f}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    
    # Test keypoint detection
    print("\n3. Testing detect_face_keypoints()...")
    try:
        face = faces[0]  # Use first detected face
        keypoints = detect_face_keypoints(face)
        print(f"   Keypoints detected - Shape: {keypoints.shape}")
        print(f"   Keypoints:\n{keypoints}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    
    # Test face alignment
    print("\n4. Testing face alignment...")
    try:
        # We need to use _align_face helper or manually call warp_face
        # For simplicity, let's calculate transformation and warp
        reference_keypoints = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        transformation_matrix = cv2.estimateAffinePartial2D(
            keypoints.astype(np.float32),
            reference_keypoints,
            method=cv2.LMEDS
        )[0]
        
        aligned_face = warp_face(image, transformation_matrix)
        print(f"   Face aligned - Shape: {aligned_face.shape}")
        
        # Save aligned face for inspection
        cv2.imwrite("aligned_face.jpg", aligned_face)
        print(f"   Aligned face saved as 'aligned_face.jpg'")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    
    # Test compute_face_embedding (MODE 1: using Face object)
    print("\n4. Testing compute_face_embedding() - Mode 1 (Face object)...")
    try:
        embedding = compute_face_embedding(face)
        print(f"   Embedding computed - Shape: {embedding.shape}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    
    # Test anti-spoofing (if implemented)
    print("\n6. Testing antispoof_check()...")
    try:
        spoof_score = antispoof_check(aligned_face)
        print(f"   Anti-spoof score: {spoof_score:.4f}")
        
        if spoof_score > 0.7:
            print(f"   Likely REAL face")
        elif spoof_score > 0.4:
            print(f"   Uncertain")
        else:
            print(f"   Possibly FAKE face")
        
    except NotImplementedError:
        print(f"   Anti-spoofing not implemented yet (optional)")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    
    print("\n" + "=" * 60)
    print("ALL INDIVIDUAL TESTS PASSED!")
    print("=" * 60)


def test_full_pipeline():
    """Test the complete face similarity pipeline."""
    
    print("\n\n" + "=" * 60)
    print("TESTING FULL PIPELINE")
    print("=" * 60)
    
    # Test with two images
    image1_path = "media/face2.jpg"  # Change to your first image
    image2_path = "media/face1.jpg"  # Change to your second image
    
    print(f"\nComparing:")
    print(f"  Image A: {image1_path}")
    print(f"  Image B: {image2_path}")
    
    # Read images as bytes (simulating API upload)
    try:
        with open(image1_path, 'rb') as f:
            image_a_bytes = f.read()
        
        with open(image2_path, 'rb') as f:
            image_b_bytes = f.read()
        
        print(f"\nImages loaded - Sizes: {len(image_a_bytes)} bytes, {len(image_b_bytes)} bytes")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find image file")
        print(f"  Make sure both {image1_path} and {image2_path} exist")
        return
    
    
    # Calculate similarity
    print("\nCalculating similarity...")
    try:
        similarity = calculate_face_similarity(image_a_bytes, image_b_bytes)
        
        print(f"\n{'=' * 60}")
        print(f"SIMILARITY SCORE: {similarity:.4f}")
        print(f"{'=' * 60}")
        
        # Interpret the result
        if similarity > 0.7:
            print("HIGH similarity - Likely the SAME person")
        elif similarity > 0.5:
            print("MODERATE similarity - Possibly the same person")
        elif similarity > 0.3:
            print("LOW similarity - Probably DIFFERENT people")
        else:
            print("VERY LOW similarity - Definitely DIFFERENT people")
        
    except ValueError as e:
        print(f"\nValueError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Initialize InsightFace (needed for testing without FastAPI)
    print("Initializing InsightFace models...")
    print("(First run will download models - this may take a few minutes)")
    
    import insightface
    import models
    
    models.face_detector = insightface.app.FaceAnalysis(  # Use models.face_detector
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )
    models.face_detector.prepare(ctx_id=0, det_size=(640, 640))
    
    print("Models initialized!\n")
    
    # Run tests
    test_individual_functions()
    test_full_pipeline()