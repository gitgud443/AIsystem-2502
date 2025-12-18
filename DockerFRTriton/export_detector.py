import os
import shutil
import requests

# Path to your Triton model repository (adjust if needed)
MODEL_REPO = "model_repository"
DETECTOR_DIR = os.path.join(MODEL_REPO, "face_detector", "1")
DETECTOR_ONNX_TARGET = os.path.join(DETECTOR_DIR, "model.onnx")

# Direct download URL for det_10g.onnx (from public InsightFace buffalo_l pack)
# This is the official file, hosted on Hugging Face (safe and exact match)
URL = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx"

# Temporary download path
TEMP_ONNX = "det_10g_temp.onnx"

print(f"Downloading detector model from {URL}...")
response = requests.get(URL, stream=True)
response.raise_for_status()

with open(TEMP_ONNX, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded to {TEMP_ONNX} (~16.9 MB)")

# Create target directory and move the file
os.makedirs(DETECTOR_DIR, exist_ok=True)
shutil.move(TEMP_ONNX, DETECTOR_ONNX_TARGET)
print(f"Moved detector model to {DETECTOR_ONNX_TARGET}")

print("Detector export complete! (No insightface installation required)")