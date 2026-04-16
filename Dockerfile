FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install backend deps
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy backend source (secrets excluded via .dockerignore)
COPY backend /app/backend

WORKDIR /app/backend

EXPOSE 7860

# Hugging Face Spaces uses PORT; fallback to 7860 for safety.
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]

