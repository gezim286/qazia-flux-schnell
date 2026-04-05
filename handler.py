import runpod
import torch
import base64
import os
from io import BytesIO

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    from diffusers import FluxPipeline
    print("Loading Flux Schnell model...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir="/runpod-volume/models"
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("Flux Schnell loaded.")
    return pipe

def handler(job):
    job_input = job.get("input", {})
    prompt       = job_input.get("prompt", "cinematic scene")
    aspect_ratio = job_input.get("aspect_ratio", "16:9")
    steps        = int(job_input.get("num_inference_steps", 4))
    output_fmt   = job_input.get("output_format", "jpg")

    ratio_map = {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1":  (1024, 1024),
        "4:3":  (1152, 864),
    }
    width, height = ratio_map.get(aspect_ratio, (1280, 720))

    p = load_model()
    image = p(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=0.0,
        num_inference_steps=steps,
        max_sequence_length=256,
    ).images[0]

    buf = BytesIO()
    fmt = "JPEG" if output_fmt in ("jpg", "jpeg") else "PNG"
    image.save(buf, format=fmt, quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image_b64": b64,
        "format": output_fmt,
        "width": width,
        "height": height,
    }

runpod.serverless.start({"handler": handler})
