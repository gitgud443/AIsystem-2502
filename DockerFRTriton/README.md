# Face Recognition on Triton Inference Server

### Description
- Exports FR backbone and SCRFD innate insightface detector to ONNX (I didn't use retina at all)
- Serves both models on Triton (CPU for now)
- Full pipeline: face detection -> crop -> embedding -> cosine similarity
- FastAPI wrapper with Swagger UI at /docs

### How to run :
> I used windows and visual studio code with powershell, so I modified start.sh and dockerfile accordingly.

Clone repo, then -
Go to /DockerFRTriton on visual studio then run with :

````
docker build -t fr-triton -f Docker/Dockerfile .
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 3000:3000 fr-triton
````

Then connect on the triton server with swagger UI at http://localhost:3000/docs


Thank you for everything, teacher ! this semester was very interesting. 
(Even though this final HW was a real pain to debug)
