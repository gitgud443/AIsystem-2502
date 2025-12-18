import argparse
import shutil
from pathlib import Path
import onnx


def find_insightface_model(model_name: str = "buffalo_l") -> Path:
    """
    Locate the InsightFace model directory.
    InsightFace downloads models to ~/.insightface/models/
    """
    insightface_root = Path.home() / ".insightface" / "models" / model_name
    
    if not insightface_root.exists():
        raise FileNotFoundError(
            f"InsightFace model '{model_name}' not found at {insightface_root}.\n"
            f"Please run your InsightFace code once to download the models, or specify a different path."
        )
    
    return insightface_root


def make_output_dynamic(model_path: Path) -> None:
    """
    Make the output batch dimension dynamic for Triton compatibility.
    
    InsightFace models often have fixed output shapes like [1, 512].
    Triton requires dynamic batch dimensions [-1, 512] for batching support.
    """
    model = onnx.load(str(model_path))
    
    modified = False
    for output in model.graph.output:
        # Get the first dimension (batch dimension)
        if len(output.type.tensor_type.shape.dim) > 0:
            first_dim = output.type.tensor_type.shape.dim[0]
            
            # Check if it's a fixed dimension
            if first_dim.HasField('dim_value') and first_dim.dim_value == 1:
                print(f"[convert] Output '{output.name}' has fixed batch size: {first_dim.dim_value}")
                print(f"[convert] Changing to dynamic batch dimension...")
                
                # Clear the fixed value and set dynamic parameter
                first_dim.Clear()
                first_dim.dim_param = "batch"
                
                modified = True
                print(f"[convert] ✓ Output '{output.name}' now has dynamic batch dimension")
    
    if modified:
        # Save the modified model
        onnx.save(model, str(model_path))
        print(f"[convert] ✓ Saved model with dynamic output dimensions")
    else:
        print(f"[convert] Output already has dynamic batch dimension (no changes needed)")


def extract_recognition_model(model_dir: Path, output_path: Path) -> None:
    """
    Extract the face recognition (ArcFace) ONNX model from InsightFace buffalo_l.
    The FR model is typically named 'w600k_r50.onnx' or similar.
    """
    # Common FR model names in buffalo_l
    possible_names = [
        "w600k_r50.onnx",      # Most common in buffalo_l
        "w600k_mbf.onnx",       # MobileFaceNet variant
        "glintr100.onnx",       # R100 variant
    ]
    
    fr_model_path = None
    for name in possible_names:
        candidate = model_dir / name
        if candidate.exists():
            fr_model_path = candidate
            print(f"[convert] Found FR model: {name}")
            break
    
    if fr_model_path is None:
        # List available models to help user
        available = list(model_dir.glob("*.onnx"))
        raise FileNotFoundError(
            f"Could not find FR model in {model_dir}.\n"
            f"Available ONNX models: {[f.name for f in available]}\n"
            f"Please check the model directory or specify the correct model name."
        )
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the ONNX model
    shutil.copy2(fr_model_path, output_path)
    
    # Make output batch dimension dynamic for Triton compatibility
    print(f"[convert] Making output batch dimension dynamic...")
    make_output_dynamic(output_path)
    
    # Verify the model
    try:
        onnx.checker.check_model(onnx.load(str(output_path)))
        print(f"[convert] ✓ Successfully extracted FR model to: {output_path}")
        print(f"[convert] ✓ Model validation passed")
    except Exception as e:
        print(f"[convert] ⚠ Warning: Model validation failed: {e}")
        print(f"[convert] The model was still copied, but may have issues.")


def print_model_info(model_path: Path) -> None:
    """Print basic information about the ONNX model."""
    try:
        model = onnx.load(str(model_path))
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        
        # Input info
        print("\nInputs:")
        for input_tensor in model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else d.dim_param if d.dim_param else "dynamic" 
                   for d in input_tensor.type.tensor_type.shape.dim]
            print(f"  - {input_tensor.name}: {dims}")
        
        # Output info
        print("\nOutputs:")
        for output_tensor in model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else d.dim_param if d.dim_param else "dynamic" 
                   for d in output_tensor.type.tensor_type.shape.dim]
            print(f"  - {output_tensor.name}: {dims}")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Could not read model info: {e}")


def convert_model_to_onnx(
    insightface_model_name: str,
    onnx_path: Path,
    custom_model_dir: Path = None
) -> None:
    """
    Extract the InsightFace FR model (ArcFace backbone) to ONNX for Triton.
    
    Args:
        insightface_model_name: Name of InsightFace model pack (e.g., 'buffalo_l')
        onnx_path: Destination path for the ONNX model
        custom_model_dir: Optional custom directory containing InsightFace models
    """
    print(f"[convert] Extracting InsightFace model: {insightface_model_name}")
    
    # Find the InsightFace model directory
    if custom_model_dir and custom_model_dir.exists():
        model_dir = custom_model_dir
        print(f"[convert] Using custom model directory: {model_dir}")
    else:
        model_dir = find_insightface_model(insightface_model_name)
        print(f"[convert] Found InsightFace models at: {model_dir}")
    
    # Extract the recognition model
    extract_recognition_model(model_dir, onnx_path)
    
    # Print model information
    print_model_info(onnx_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract InsightFace FR model to ONNX for Triton."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="buffalo_l",
        help="InsightFace model pack name (default: buffalo_l)",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("model_repository/fr_model/1/model.onnx"),
        help="Destination for exported ONNX file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Custom directory containing InsightFace models (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_model_to_onnx(
        args.model_name,
        args.onnx_path,
        args.model_dir
    )


if __name__ == "__main__":
    main()