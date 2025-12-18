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
FR_INPUT_NAME = "input"
FR_OUTPUT_NAME = "embedding"
FR_IMAGE_SIZE = (112, 112)

# Face Detector Model (SCRFD 10G)
DET_MODEL_NAME = "face_detector"
DET_INPUT_NAME = "input"  # Standard for most detectors
# Outputs: SCRFD produces multiple heads, but we typically use the concatenated ones
# Common outputs for det_10g.onnx: score_0, bbox_0, score_1, bbox_1, ..., score_4, bbox_4
# We'll request the first few and let postprocessing handle strides
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
    fr_config_text = textwrap.dedent(f"""
        name: "{FR_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        input [
          {{
            name: "{FR_INPUT_NAME}"
            data_type: TYPE_FP32
            dims: [3, {FR_IMAGE_SIZE[0]}, {FR_IMAGE_SIZE[1]}]
          }}
        ]
        output [
          {{
            name: "{FR_OUTPUT_NAME}"
            data_type: TYPE_FP32
            dims: [512]
          }}
        ]
        instance_group [ {{ kind: KIND_CPU }} ]
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
    det_config_text = textwrap.dedent(f"""
        name: "{DET_MODEL_NAME}"
        platform: "onnxruntime_onnx"
        max_batch_size: 1
        input [
          {{
            name: "{DET_INPUT_NAME}"
            data_type: TYPE_FP32
            format: FORMAT_NCHW
            dims: [3, -1, -1]  # Dynamic height/width
          }}
        ]
        dynamic_batching {{ }}
    """).strip()

    # Add all expected outputs
    output_block = "\n".join(
        f'''
        {{
          name: "{name}"
          data_type: TYPE_FP32
          dims: [-1, -1]
        }}''' for name in DET_OUTPUT_NAMES
    )
    det_config_text += "\noutput [" + output_block + "\n]"
    det_config_text += "\ninstance_group [ { kind: KIND_CPU } ]\n"
    det_config_path.write_text(det_config_text)

    print(f"[triton] Prepared model repository with {FR_MODEL_NAME} and {DET_MODEL_NAME}")


def start_triton_server(model_repo: Path) -> Any:
    # (unchanged — same as before)
    triton_bin = subprocess.run(["which", "tritonserver"], capture_output=True, text=True).stdout.strip()
    if not triton_bin:
        raise RuntimeError("Could not find `tritonserver` binary in PATH.")
    cmd = [
        triton_bin,
        f"--model-repository={model_repo}",
        f"--http-port={TRITON_HTTP_PORT}",
        f"--grpc-port={TRITON_GRPC_PORT}",
        f"--metrics-port={TRITON_METRICS_PORT}",
        "--allow-http=true",
        "--allow-grpc=true",
        "--allow-metrics=true",
        "--log-verbose=1",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[triton] Starting Triton server...")
    time.sleep(5)  # Give more time for two models to load
    return process


def stop_triton_server(server_handle: Any) -> None:
    # (unchanged)
    if server_handle is None:
        return
    server_handle.terminate()
    try:
        server_handle.wait(timeout=10)
        print("[triton] Triton server stopped.")
    except subprocess.TimeoutExpired:
        server_handle.kill()
        print("[triton] Triton server killed.")


def create_triton_client(url: str = f"localhost:{TRITON_HTTP_PORT}") -> Any:
    try:
        from tritonclient import http as httpclient
    except ImportError as exc:
        raise RuntimeError("tritonclient[http] required") from exc
    client = httpclient.InferenceServerClient(url=url, verbose=False)
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
    img = cropped_face.resize(FR_IMAGE_SIZE)
    np_img = np.asarray(img, dtype=np.float32) / 255.0
    np_img = np.transpose(np_img, (2, 0, 1))
    return np.expand_dims(np_img, axis=0)


def run_recognition_inference(client: Any, preprocessed_face: np.ndarray) -> np.ndarray:
    """Run inference on fr_model"""
    from tritonclient import http as httpclient

    infer_input = httpclient.InferInput(FR_INPUT_NAME, preprocessed_face.shape, "FP32")
    infer_input.set_data_from_numpy(preprocessed_face)

    infer_output = httpclient.InferRequestedOutput(FR_OUTPUT_NAME)

    response = client.infer(
        model_name=FR_MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    return response.as_numpy(FR_OUTPUT_NAME)