# ── Base image — CUDA 11.8 + Python 3.10 ──────────────────────────────────────
FROM runpod/base:0.4.0-cuda11.8.0

# System deps needed by OpenCV / GFPGAN
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Copy handler ───────────────────────────────────────────────────────────────
COPY handler.py .

# ── Pre-download model weights at build time (baked into image) ───────────────
# This removes the cold-start weight-download overhead entirely.
RUN python3 -c "\
import os, urllib.request; \
os.makedirs('weights', exist_ok=True); \
urllib.request.urlretrieve( \
    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', \
    'weights/GFPGANv1.4.pth'); \
urllib.request.urlretrieve( \
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', \
    'weights/RealESRGAN_x2plus.pth'); \
print('Weights downloaded.')"

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK NONE

# ── Start handler on container launch ─────────────────────────────────────────
CMD ["python3", "-u", "handler.py"]
