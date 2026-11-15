# Container for the public demo (Hugging Face Spaces, Docker SDK) - also runs anywhere Docker runs.
#
#   docker build -t docqa .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... docqa      # or ANTHROPIC_API_KEY=...
#   -> http://localhost:7860
#
# Without any API key the app still works: answers come from the offline extractive fallback.
# See docs/DEPLOYMENT.md.

FROM python:3.11-slim

# Hugging Face Spaces runs containers as uid 1000; everything the app writes must be owned by it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch (~200 MB instead of ~2 GB of CUDA wheels), then the project requirements.
COPY --chown=user requirements.txt .
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --user -r requirements.txt

# Download the embedding model at build time so the first visitor does not wait for it.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
RUN mkdir -p data/uploads models reports

EXPOSE 7860
# XSRF protection is off because Spaces serves the app inside an iframe on another origin,
# where Streamlit's XSRF check rejects file uploads (HTTP 403).
CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
