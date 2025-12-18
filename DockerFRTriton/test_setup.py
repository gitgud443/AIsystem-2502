#!/usr/bin/env python3
"""
Test script to verify the model export and Triton setup.
Run this after exporting models to check everything is ready.
"""

import sys
from pathlib import Path
import onnx


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists and print status."""
    if filepath.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} MISSING: {filepath}")
        return False


def check_onnx_model(model_path: Path, model_name: str) -> bool:
    """Check if ONNX model is valid."""
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        print(f"\n  {model_name} Details:")
        print(f"  Inputs:")
        for input_tensor in model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" 
                   for d in input_tensor.type.tensor_type.shape.dim]
            print(f"    - {input_tensor.name}: {dims}")
        
        print(f"  Outputs:")
        for output_tensor in model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" 
                   for d in output_tensor.type.tensor_type.shape.dim]
            print(f"    - {output_tensor.name}: {dims}")
        
        return True
    except Exception as e:
        print(f"✗ {model_name} validation FAILED: {e}")
        return False


def main():
    """Run all checks."""
    print("="*70)
    print("TRITON MODEL REPOSITORY VERIFICATION")
    print("="*70)
    
    all_passed = True
    
    # Check directory structure
    print("\n1. Checking Directory Structure...")
    print("-" * 70)
    
    model_repo = Path("model_repository")
    
    checks = [
        (model_repo, "Model repository root"),
        (model_repo / "fr_model", "FR model directory"),
        (model_repo / "fr_model" / "1", "FR model version directory"),
        (model_repo / "face_detector", "Detector directory"),
        (model_repo / "face_detector" / "1", "Detector version directory"),
    ]
    
    for path, desc in checks:
        if not check_file_exists(path, desc):
            all_passed = False
    
    # Check ONNX models
    print("\n2. Checking ONNX Models...")
    print("-" * 70)
    
    fr_model = model_repo / "fr_model" / "1" / "model.onnx"
    detector_model = model_repo / "face_detector" / "1" / "model.onnx"
    
    if check_file_exists(fr_model, "FR model ONNX"):
        if not check_onnx_model(fr_model, "FR Model"):
            all_passed = False
    else:
        all_passed = False
    
    print()
    
    if check_file_exists(detector_model, "Detector model ONNX"):
        if not check_onnx_model(detector_model, "Detector Model"):
            all_passed = False
    else:
        all_passed = False
    
    # Check config files
    print("\n3. Checking Config Files...")
    print("-" * 70)
    
    fr_config = model_repo / "fr_model" / "config.pbtxt"
    detector_config = model_repo / "face_detector" / "config.pbtxt"
    
    if not check_file_exists(fr_config, "FR model config"):
        print("   Note: Config will be auto-generated when server starts")
    
    if not check_file_exists(detector_config, "Detector config"):
        print("   Note: Config will be auto-generated when server starts")
    
    # Check InsightFace models directory
    print("\n4. Checking InsightFace Source...")
    print("-" * 70)
    
    insightface_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    if check_file_exists(insightface_dir, "InsightFace buffalo_l directory"):
        print(f"   Available models:")
        for onnx_file in insightface_dir.glob("*.onnx"):
            print(f"     - {onnx_file.name}")
    else:
        print("   ⚠ InsightFace models not found. Run InsightFace once to download.")
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou're ready to run the server:")
        print("  python run_fastapi.py")
        print("\nOr build Docker:")
        print("  docker build -t fr-triton -f Docker/Dockerfile .")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nTo fix:")
        print("  1. Export FR model:")
        print("     python convert_to_onnx.py")
        print("  2. Export detector:")
        print("     python export_detector.py")
        print("  3. Run this test again")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())