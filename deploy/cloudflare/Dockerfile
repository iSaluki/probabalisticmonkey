# Container image for Probabilistic Monkey on Cloudflare Containers.
# Built by `wrangler deploy` with the repository root as the build context.
FROM python:3.12-slim

# model (default) or model_big
ARG MODEL_DIR=model

WORKDIR /monkey
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY app/ app/
COPY ${MODEL_DIR}/ model/

EXPOSE 8080
CMD ["python", "app/server.py", "--host", "0.0.0.0", "--port", "8080", "--no-browser"]
