"""GPU inference service. Install GPU requirements and run on a CUDA machine.
This baseline intentionally uses SDXL Inpainting only; ControlNet/LoRA are not silently loaded.
"""
import io, os, threading, inspect
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from diffusers import StableDiffusionXLInpaintPipeline

MODEL_ID = os.getenv("INPAINT_MODEL_ID", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
PIPE = None
GROUNDING_PROCESSOR = GROUNDING_MODEL = SAM_PROCESSOR = SAM_MODEL = None
LOCK = threading.Lock()
BACKEND_TOKEN = os.getenv("GPU_BACKEND_TOKEN", "").strip()
GROUNDING_DINO_ID = os.getenv("GROUNDING_DINO_ID", "IDEA-Research/grounding-dino-tiny")
SAM_ID = os.getenv("SAM_ID", "facebook/sam-vit-base")


def _check_token(token: str | None):
    # Enforce auth when a token is configured; supports local dev without a token.
    if BACKEND_TOKEN and (not token or not __import__("hmac").compare_digest(token, BACKEND_TOKEN)):
        raise HTTPException(401, "Missing or invalid backend token")


def _grounding_post_process(processor, outputs, input_ids, target_sizes):
    """Call Grounding DINO post-processing across Transformers API variants."""
    method = processor.post_process_grounded_object_detection
    kwargs = {"text_threshold": 0.20, "target_sizes": target_sizes}
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        params = {}
    if "box_threshold" in params or not params:
        kwargs["box_threshold"] = 0.25
    elif "threshold" in params:
        kwargs["threshold"] = 0.25
    else:
        raise RuntimeError(
            "Unsupported Grounding DINO processor API: expected 'box_threshold' or 'threshold'."
        )
    try:
        return method(outputs, input_ids, **kwargs)
    except TypeError as exc:
        # Some wrapped/decorated processor methods do not expose their true signature.
        if "box_threshold" in kwargs and "box_threshold" in str(exc):
            kwargs.pop("box_threshold")
            kwargs["threshold"] = 0.25
            return method(outputs, input_ids, **kwargs)
        if "threshold" in kwargs and "threshold" in str(exc):
            kwargs.pop("threshold")
            kwargs["box_threshold"] = 0.25
            return method(outputs, input_ids, **kwargs)
        raise


def _automatic_mask(src: Image.Image, target: str, prompt: str) -> Image.Image:
    phrase = (target or "").strip()
    if not phrase or phrase.lower() in {"object", "thing", "item", "unknown"}:
        phrase = (prompt or "").strip()
    phrase = phrase[:160].strip().rstrip(".")
    if not phrase:
        raise HTTPException(422, "Automatic masking needs a target description or a manual mask.")
    try:
        with torch.inference_mode():
            inputs = GROUNDING_PROCESSOR(images=src, text=[phrase], return_tensors="pt")
            outputs = GROUNDING_MODEL(**inputs)
            det = _grounding_post_process(
                GROUNDING_PROCESSOR, outputs, inputs.input_ids,
                target_sizes=[(src.height, src.width)],
            )[0]
        boxes, scores = det.get("boxes"), det.get("scores")
        if boxes is None or len(boxes) == 0:
            raise HTTPException(422, f"Could not locate '{phrase}' automatically. Try a clearer target or upload a mask.")
        idx = int(torch.argmax(scores).item()) if scores is not None and len(scores) else 0
        box = boxes[idx].detach().cpu().tolist()
        sam_inputs = SAM_PROCESSOR(images=src, input_boxes=[[box]], return_tensors="pt")
        with torch.inference_mode():
            sam_outputs = SAM_MODEL(**sam_inputs)
        masks = SAM_PROCESSOR.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(), sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]
        candidates = masks[0][0]
        ious = sam_outputs.iou_scores.detach().cpu()[0, 0]
        mask = candidates[int(torch.argmax(ious).item())].numpy() > 0
        if not mask.any():
            raise HTTPException(422, "Automatic segmentation returned an empty mask.")
        mask_u8 = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
        return Image.fromarray(mask_u8, mode="L").resize(src.size, Image.Resampling.NEAREST)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Automatic masking failed: {type(exc).__name__}: {str(exc)[:180]}") from exc

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPE, GROUNDING_PROCESSOR, GROUNDING_MODEL, SAM_PROCESSOR, SAM_MODEL
    if not torch.cuda.is_available(): raise RuntimeError("GPU service requires CUDA. Use a GPU-enabled host.")
    PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    PIPE.to("cuda")
    PIPE.set_progress_bar_config(disable=True)
    # Segmentation models stay on CPU to avoid competing with the diffusion model's VRAM.
    GROUNDING_PROCESSOR = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
    GROUNDING_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID).to("cpu").eval()
    SAM_PROCESSOR = SamProcessor.from_pretrained(SAM_ID)
    SAM_MODEL = SamModel.from_pretrained(SAM_ID).to("cpu").eval()
    yield
    PIPE = GROUNDING_PROCESSOR = GROUNDING_MODEL = SAM_PROCESSOR = SAM_MODEL = None

app = FastAPI(title="SDXL Inpainting GPU Service", version="0.2.0", lifespan=lifespan)

@app.get("/health")
def health(x_backend_token: str | None = Header(default=None)):
    _check_token(x_backend_token)
    return {"status":"ok" if PIPE is not None else "loading", "model":MODEL_ID, "cuda":torch.cuda.is_available(), "automatic_masking": GROUNDING_MODEL is not None and SAM_MODEL is not None}

@app.post("/v1/edit")
def edit(image: UploadFile = File(...), mask: UploadFile | None = File(None), prompt: str = Form(...), operation: str = Form("modify"), target: str = Form("object"), auto_mask: bool = Form(True), x_backend_token: str | None = Header(default=None), negative_prompt: str = Form("blurry, distorted, low quality, artifacts"), steps: int = Form(30), guidance_scale: float = Form(7.0), strength: float = Form(0.99), seed: int = Form(42)):
    _check_token(x_backend_token)
    if PIPE is None: raise HTTPException(503, "Model is not loaded yet.")
    if operation not in {"add", "remove", "modify", "replace"}: raise HTTPException(400, "Unsupported operation")
    try:
        original = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        if mask is not None:
            mask_image = Image.open(io.BytesIO(mask.file.read())).convert("L")
        elif auto_mask:
            mask_image = _automatic_mask(original, target, prompt)
        else:
            raise HTTPException(422, "No mask supplied. Upload a mask or enable automatic masking.")
    except HTTPException:
        raise
    except Exception as exc: raise HTTPException(400, f"Invalid image or mask: {exc}")
    if original.size != mask_image.size: mask_image = mask_image.resize(original.size, Image.Resampling.NEAREST)
    # White means editable, black means preserve. Binarize mask for predictable inpainting.
    mask_image = mask_image.point(lambda p: 255 if p >= 128 else 0)
    if mask_image.getbbox() is None: raise HTTPException(400, "Mask is empty; white pixels must mark the editable region.")
    w, h = original.size
    # Preserve aspect ratio; SDXL dimensions must be divisible by 8. Avoid
    # forcing both axes to 512, which would distort portrait/landscape images.
    scale = min(1024 / max(w, h), 1.0)
    tw = max(8, int(round((w * scale) / 8) * 8))
    th = max(8, int(round((h * scale) / 8) * 8))
    with LOCK:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        try:
            edited = PIPE(prompt=prompt, negative_prompt=negative_prompt, image=original.resize((tw,th), Image.Resampling.LANCZOS), mask_image=mask_image.resize((tw,th), Image.Resampling.NEAREST), num_inference_steps=max(10,min(60,steps)), guidance_scale=max(1.0,min(12.0,guidance_scale)), strength=max(0.1,min(1.0,strength)), generator=generator).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); raise HTTPException(507, "GPU out of memory. Try a smaller image or lower inference resolution.")
    edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    # Hard composite guarantees original pixels outside the user-provided mask are preserved.
    final = Image.composite(edited, original, mask_image)
    buf = io.BytesIO(); final.save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={"Content-Disposition":"attachment; filename=edited.png", "X-Mask-Mode": "provided" if mask is not None else "automatic"})
