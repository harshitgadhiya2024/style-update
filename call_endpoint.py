"""
Call your deployed RunPod deblur endpoint.

Usage:
    python call_endpoint.py --endpoint YOUR_ENDPOINT_ID --image photo.jpg
    python call_endpoint.py --endpoint YOUR_ENDPOINT_ID --url https://...
"""

import argparse
import base64
import os

import requests


RUNPOD_API_KEY  = os.getenv("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")


def run_job(endpoint_id: str, payload: dict) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json={"input": payload}, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="RunPod Endpoint ID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a local image file")
    group.add_argument("--url",    help="Public URL of the image")
    parser.add_argument("--out",   default="output_sharp.png", help="Output file path")
    args = parser.parse_args()

    if args.image:
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {"image_base64": b64}
    else:
        payload = {"image_url": args.url}

    print(f"Submitting job to endpoint {args.endpoint} ...")
    result = run_job(args.endpoint, payload)

    output = result.get("output", {})
    if "error" in output:
        print(f"Error: {output['error']}")
        return

    b64_out = output.get("image_base64", "")
    if not b64_out:
        print("No image returned:", result)
        return

    img_bytes = base64.b64decode(b64_out)
    with open(args.out, "wb") as f:
        f.write(img_bytes)

    w, h = output.get("width"), output.get("height")
    print(f"✓ Saved sharp image → {args.out}  ({w}×{h})")


if __name__ == "__main__":
    main()
