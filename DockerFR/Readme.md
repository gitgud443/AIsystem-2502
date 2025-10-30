## Face Recognition API Project

This FastAPI project scaffolds the major pieces of a face recognition pipeline for students. Your job is to implement the missing logic in `util.py` so that the `/face-similarity` endpoint can compare two uploaded face images.

### What You Need to Implement
- `detect_faces(image)` — locate faces in the raw image bytes.
- `compute_face_embedding(face_image)` — generate a numerical embedding per face.
- `detect_face_keypoints(face_image)` — find facial landmarks to support alignment.
- `warp_face(image, homography_matrix)` — warp or align faces with the estimated homography.
- `antispoof_check(face_image)` — score the likelihood that a face is real.
- `calculate_face_similarity(image_a, image_b)` — orchestrate the full pipeline: detect, align, spoof-check, embed, and output a similarity score.

All functions currently raise `NotImplementedError`; replace each with your own implementation and add any supporting helpers you need.

### API Overview
- `GET /health` — simple service liveness check.
- `GET /torch-version` — returns the underlying PyTorch version if installed.
- `POST /face-similarity` — accepts two uploaded images (`image_a`, `image_b`) and responds with a similarity score once the utilities are implemented. Until then, the endpoint returns HTTP 501.

Swagger UI is automatically generated at `http://localhost:5003/docs` (ReDoc at `/redoc`) when the service is running.

### Submission Guidelines
- Project due date: **November 16**.
- Submit your work via a Git pull request named `submit_<studentID>_<name>` (for example, `submit_20230001_AlexKim`).

## Basic Docker Workflow

Ensure Docker Desktop (or another Docker engine) is running before you begin.

- Build the image  
  `docker build -t fr-api -f Docker/Dockerfile .`
- Run the container with a friendly name  
  `docker run -d --name fr-container -p 5003:5000 fr-api`
- Check running containers (optional)  
  `docker ps`
- Tail logs (optional)  
  `docker logs -f fr-container`
- Open the FastAPI Swagger UI  
  `http://localhost:5003/docs`
- Stop and remove the container when finished  
  `docker stop fr-container && docker rm fr-container`
