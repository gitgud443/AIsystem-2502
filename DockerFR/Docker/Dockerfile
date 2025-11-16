FROM python:3.9-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir numpy==2.0.2
RUN pip install --no-cache-dir onnxruntime==1.15.1
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir insightface==0.7.3
RUN pip install --no-cache-dir fastapi==0.104.1
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir uvicorn[standard]==0.24.0
RUN pip install --no-cache-dir python-multipart==0.0.6
RUN pip install --no-cache-dir torch

# Copy application files
COPY main.py util.py models.py ./

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]