import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Dict

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
DET_INPUT_NAME = "input"
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


def create_triton_client(url: str = "triton:8000") -> Any:  # Default for Docker
    try:
        from tritonclient.http import InferenceServerClient as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] required") from exc

    client = httpclient(url=url, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server at {url} not live.")
    return client


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
    """Run inference on face_detector model"""
    from tritonclient import http as httpclient

    inputs = [httpclient.InferInput(DET_INPUT_NAME, preprocessed_image.shape, "FP32")]
    inputs[0].set_data_from_numpy(preprocessed_image)

    outputs = [httpclient.InferRequestedOutput(name) for name in DET_OUTPUT_NAMES]

    response = client.infer(
        model_name=DET_MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

    return {name: response.as_numpy(name) for name in DET_OUTPUT_NAMES}


def preprocess_for_recognition(cropped_face: Image.Image) -> np.ndarray:
    """Preprocess aligned face for ArcFace model"""
    img = cropped_face.resize((112, 112))
    np_img = np.asarray(img, dtype=np.float32) / 255.0
    np_img = np.transpose(np_img, (2, 0, 1))  
    np_img = np.expand_dims(np_img, axis=0)   
    return np_img


def run_recognition_inference(client: Any, preprocessed_face: np.ndarray) -> np.ndarray:
    """Run inference on fr_model"""
    from tritonclient import http as httpclient

    infer_input = httpclient.InferInput(FR_INPUT_NAME, preprocessed_face.shape, "FP32")
    infer_input.set_data_from_numpy(preprocessed_face)

    infer_output = httpclient.InferRequestedOutput("683")

    response = client.infer(
        model_name=FR_MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    return response.as_numpy(FR_OUTPUT_NAME)