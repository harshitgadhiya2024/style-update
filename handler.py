"""
RunPod Serverless Handler — Image Deblurring (GFPGAN + Real-ESRGAN)

Flow:
    Input  : { "image_url": "<public URL>" }   OR   { "image_base64": "<base64 string>" }
    Output : { "image_base64": "<base64 PNG>", "width": W, "height": H }

Environment variables (optional, override defaults):
    DEBLUR_UPSCALE      - 1 or 2  (default: 2)
    DEBLUR_FULL_IMAGE   - "true" / "false"  (default: true)
"""

import base64
import importlib
import io
import os
import subprocess
import sys
import urllib.request

import runpod


# ── CONFIG (override via env vars) ────────────────────────────────────────────
DEBLUR_UPSCALE    = 2
DEBLUR_FULL_IMAGE = True

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


# ── ONE-TIME SETUP (runs at container cold-start, not per request) ─────────────

def _ensure_packages():
    """Auto-install missing Python packages."""
    for pkg, mod in [
        ("gfpgan",    "gfpgan"),
        ("basicsr",   "basicsr"),
        ("facexlib",  "facexlib"),
        ("realesrgan","realesrgan"),
        ("opencv-python-headless", "cv2"),
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[setup] Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _patch_torchvision():
    """Shim for torchvision >= 0.16 which removed functional_tensor."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import torchvision.transforms.functional as _ftf
        sys.modules["torchvision.transforms.functional_tensor"] = _ftf


def _download_weights():
    """Download model weights once to the weights/ folder."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    files = {
        "GFPGANv1.4.pth": (
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        ),
    }
    if DEBLUR_FULL_IMAGE:
        files["RealESRGAN_x2plus.pth"] = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        )

    for fname, url in files.items():
        dest = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(dest):
            print(f"[setup] Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"[setup] Saved → {dest}")


def _build_restorer():
    """Initialise and return a GFPGANer instance (with optional Real-ESRGAN bg upsampler)."""
    from gfpgan import GFPGANer

    bg_upsampler = None
    if DEBLUR_FULL_IMAGE:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        esrgan_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x2plus.pth")
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path=esrgan_path,
            model=RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            ),
            tile=512, tile_pad=16, pre_pad=0, half=False, device="cpu",
        )
        print("[setup] Real-ESRGAN loaded")

    restorer = GFPGANer(
        model_path=os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth"),
        upscale=DEBLUR_UPSCALE,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
        device="cpu",
    )
    print("[setup] GFPGAN loaded")
    return restorer


# ── Cold-start initialisation ─────────────────────────────────────────────────
print("[setup] Starting cold-start initialisation ...")
_ensure_packages()
_patch_torchvision()
_download_weights()
RESTORER = _build_restorer()
print("[setup] ✓ Models ready — handler is hot")


# ── HANDLER ───────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    RunPod job handler.

    Accepted input formats:
        { "image_url":    "https://..." }
        { "image_base64": "<base64 string>" }

    Returns:
        { "image_base64": "<base64 PNG>", "width": W, "height": H }
    """
    import cv2
    import numpy as np
    import requests

    job_input = job.get("input", {})

    # ── 1. Load image ──────────────────────────────────────────────────────────
    if "image_url" in job_input:
        url = job_input["image_url"]
        print(f"[handler] Fetching image from URL: {url[:80]}")
        resp = requests.get(url, timeout=(10, 30))
        resp.raise_for_status()
        img_bytes = resp.content

    elif "image_base64" in job_input:
        print("[handler] Decoding base64 image ...")
        raw = job_input["image_base64"]
        # Strip data-URI prefix if present (e.g. "data:image/png;base64,...")
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)

    else:
        return {"error": "Provide either 'image_url' or 'image_base64' in the input."}

    # Decode bytes → OpenCV BGR array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image. Make sure it is a valid PNG/JPEG."}

    h, w = img.shape[:2]
    print(f"[handler] Image loaded: {w}×{h}")

    # ── 2. Run GFPGAN + Real-ESRGAN ────────────────────────────────────────────
    print("[handler] Running deblur / restoration ...")
    _, _, sharp = RESTORER.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5,
    )

    sh, sw = sharp.shape[:2]
    print(f"[handler] Output size: {sw}×{sh}")

    # ── 3. Encode result as base64 PNG ─────────────────────────────────────────
    success, buf = cv2.imencode(".png", sharp)
    if not success:
        return {"error": "Failed to encode output image as PNG."}

    b64_result = base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "image_base64": b64_result,
        "width":        sw,
        "height":       sh,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
