# Container image for Probabilistic Monkey and Probabilistic Gorilla on Cloudflare Containers.
# Built by `wrangler deploy` with the repository root as the build context; the
# root .dockerignore limits the context to app/, model/ and model_gorilla/.
FROM python:3.12-slim

WORKDIR /primates
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
# app/ plus every trained model folder that exists (model_gorilla/ only once it is trained)
COPY . .

EXPOSE 8080
CMD ["python", "app/server.py", "--host", "0.0.0.0", "--port", "8080", "--no-browser"]
