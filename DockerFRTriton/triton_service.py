import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Dict

import time
from tritonclient.http import InferenceServerClient
from tritonclient.utils import InferenceServerException
import numpy as np
from io import BytesIO
from PIL import Image

# === Model Constants ===
# Face Recognition Model (ArcFace)
FR_MODEL_NAME = "fr_model"
FR_INPUT_NAME = "input.1"   
FR_OUTPUT_NAME = "683" 
FR_MODEL_IMAGE_SIZE = (112, 112)

# Face Detector Model (SCRFD 10G)
DET_MODEL_NAME = "face_detector"
# DET_INPUT_NAME = "input"
DET_OUTPUT_NAMES = [f"score_{i}" for i in range(5)] + [f"bbox_{i}" for i in range(5)]

TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002


def prepare_model_repository(model_repo: Path) -> None:
    """
    Populate the Triton model repository with both models and their config.pbtxt files.
    """
    # --- Face Recognition Model ---
    fr_model_dir = model_repo / FR_MODEL_NAME / "1"
    fr_model_path = fr_model_dir / "model.onnx"
    fr_config_path = model_repo / FR_MODEL_NAME / "config.pbtxt"

    if not fr_model_path.exists():
        raise FileNotFoundError(
            f"Missing FR model at {fr_model_path}. Run convert_to_onnx.py first."
        )

    fr_model_dir.mkdir(parents=True, exist_ok=True)
    fr_config_text = textwrap.dedent("""
    name: "fr_model"
    platform: "onnxruntime_onnx"
    max_batch_size: 1
    input [
      {
        name: "input.1"
        data_type: TYPE_FP32
        dims: [3, 112, 112]
      }
    ]
    output [
      {
        name: "683"
        data_type: TYPE_FP32
        dims: [512]
      }
    ]
    instance_group [ { kind: KIND_CPU } ]
""").strip() + "\n"
    fr_config_path.write_text(fr_config_text)

    # --- Face Detector Model ---
    det_model_dir = model_repo / DET_MODEL_NAME / "1"
    det_model_path = det_model_dir / "model.onnx"
    det_config_path = model_repo / DET_MODEL_NAME / "config.pbtxt"

    if not det_model_path.exists():
        raise FileNotFoundError(
            f"Missing detector model at {det_model_path}. Run export_detector.py first."
        )

    det_model_dir.mkdir(parents=True, exist_ok=True)

    det_config_text = textwrap.dedent("""
        name: "face_detector"
        platform: "onnxruntime_onnx"
        max_batch_size: 1
        input [
          {
            name: "input"
            data_type: TYPE_FP32
            dims: [ 3, -1, -1 ]
          }
        ]
        dynamic_batching { }
        instance_group [ { kind: KIND_CPU } ]
    """).strip() + "\n"

    # Build outputs as a single comma-separated string
    output_entries = []
    for i in range(5):
        output_entries.append(
            '{ name: "score_' + str(i) + '" data_type: TYPE_FP32 dims: [ -1, 1 ] }'
        )
        output_entries.append(
            '{ name: "bbox_' + str(i) + '" data_type: TYPE_FP32 dims: [ -1, 4 ] }'
        )

    det_config_text += "output [ " + " ".join(output_entries) + " ]\n"
    det_config_path.write_text(det_config_text)

    print(f"[triton] Prepared model repository with {FR_MODEL_NAME} and {DET_MODEL_NAME}")


def create_triton_client(url: str = "localhost:8000", max_retries: int = 30, delay: int = 2):
    """Create Triton client with retry until server is live."""
    for attempt in range(max_retries):
        try:
            client = InferenceServerClient(url=url, verbose=False)
            if client.is_server_live():
                print(f"[Triton] Connected successfully to {url}")
                return client
            else:
                print(f"[Triton] Server not live yet (attempt {attempt+1}/{max_retries})...")
        except (InferenceServerException, ConnectionRefusedError, Exception) as e:
            print(f"[Triton] Connection attempt {attempt+1}/{max_retries} failed: {e}")
        
        time.sleep(delay)
    
    raise RuntimeError(f"Failed to connect to Triton server at {url} after {max_retries} attempts")


def preprocess_for_detector(image_bytes: bytes, target_size=(640, 640)) -> np.ndarray:
    """Resize and normalize image for SCRFD detector"""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    np_img = np.asarray(img, dtype=np.float32)
    np_img = np_img / 255.0
    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    np_img = np.expand_dims(np_img, axis=0)  # Add batch dim
    return np_img


def run_detector_inference(client: Any, preprocessed_image: np.ndarray) -> Dict[str, np.ndarray]:
    """Run inference on face_detector model and return all outputs by name"""
    from tritonclient.http import InferInput, InferRequestedOutput
    
    # Dynamically get model metadata
    metadata = client.get_model_metadata(model_name=DET_MODEL_NAME)
    
    # Get the ACTUAL input name (there should be only one)
    input_name = metadata['inputs'][0]['name']
    output_names = [out['name'] for out in metadata['outputs']]
    
    print(f"[Detector] Using input name: {input_name}")
    print(f"[Detector] Output names: {output_names}")
    
    # Create input with the correct name
    inputs = [InferInput(input_name, preprocessed_image.shape, "FP32")]
    inputs[0].set_data_from_numpy(preprocessed_image)
    
    outputs = [InferRequestedOutput(name) for name in output_names]
    
    response = client.infer(
        model_name=DET_MODEL_NAME,
        inputs=inputs,
        outputs=outputs
    )
    
    result_dict = {name: response.as_numpy(name) for name in output_names}

    print(f"[Detector] Raw output shapes: { {k: v.shape for k, v in result_dict.items()} }")
    return result_dict


def preprocess_for_recognition(cropped_face: Image.Image) -> np.ndarray:
    img = cropped_face.resize(FR_MODEL_IMAGE_SIZE)
    np_img = np.asarray(img, dtype=np.float32)
    np_img = (np_img - 127.5) / 128.0   # Standard ArcFace norm
    np_img = np.transpose(np_img, (2, 0, 1))  # CHW
    np_img = np.expand_dims(np_img, axis=0)   # batch
    return np_img


def run_recognition_inference(client: Any, preprocessed_face: np.ndarray) -> np.ndarray:
    """Run inference on fr_model"""
    from tritonclient.http import InferInput, InferRequestedOutput
    
    infer_input = InferInput(FR_INPUT_NAME, preprocessed_face.shape, "FP32")
    infer_input.set_data_from_numpy(preprocessed_face)
    
    infer_output = InferRequestedOutput(FR_OUTPUT_NAME)
    
    response = client.infer(
        model_name=FR_MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    
    embedding = response.as_numpy(FR_OUTPUT_NAME)
    return embedding[0]  # remove batch dimension