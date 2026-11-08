FROM python:3.12

WORKDIR /app

# ── Layer 0: Tesseract OCR for scanned PDFs (English, Marathi, Hindi) ─────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-mar tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: install deps (cached until requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: backend package ───────────────────────────────────────────────────
COPY backend/ backend/

# ── Layer 3: documents ─────────────────────────────────────────────────────────
COPY documents/ documents/

# ── Layer 4: application files (change most often — last for cache efficiency) ─
COPY app.py .
COPY evaluate.py .
COPY main.py .
COPY goldens.json .
COPY .streamlit/ .streamlit/
COPY assets/ assets/

# Rebrand Streamlit's loading page (tab title + favicon before the app runs)
COPY scripts/ scripts/
RUN python scripts/brand_streamlit.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
