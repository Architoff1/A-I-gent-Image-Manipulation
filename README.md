# Image Agent — conversational local app + Kaggle GPU backend

This release adds persistent conversations and image-version records to the working local Streamlit/FastAPI/LangGraph app. Image inference remains on the separately running authenticated Kaggle GPU service.

## Current features

- Chat-style Streamlit interface with a conversation sidebar, new chat, rename and delete.
- SQLite persistence for conversations, messages, and image versions (`data/image_agent.sqlite3`).
- Current-image preview overlay and download.
- Immutable image-version records, version downloads, and restore-to-current support.
- Continue a conversation from its current image without uploading it again.
- Optional source image, reference image, and mask attachments.
- Existing local LangGraph/Ollama planning and authenticated HTTPS call to the Kaggle GPU endpoint.
- Optional automatic masking: local requests set `auto_mask=true` unless a manual mask is attached.
- The first source image is saved as the conversation original before inference, so failed edits can be retried without losing the source.
- Image uploads are limited to 25 MB and file access is restricted to known app upload/output folders.
- File serving resolves individual image files by basename across the configured and project-default storage folders, returning a clean 404 rather than passing a directory to `FileResponse`.
- Relative output paths are resolved against the project directory, not whichever directory happens to be the shell's current working directory.

## Important capability boundaries

This is an integration release, not a claim that every requested AI feature is complete:

- **Automatic masking:** when no manual mask is provided, the Kaggle endpoint attempts Grounding DINO + SAM segmentation. A supplied white-on-black mask takes precedence (white = editable; black = preserve). If detection fails, the backend returns a clear error rather than pretending the edit succeeded.
- **Reference-image conditioning is not connected to the diffusion model.** Reference images are stored with the message and shown in the conversation, but the current GPU pipeline does not use their pixels to condition generation. The API explicitly notes this limitation rather than pretending the reference was applied.
- Automatic segmentation can select the wrong object when multiple instances are present. Manual masks remain the fallback for ambiguous targets.
- The existing generation and analysis nodes remain placeholders.
- The currently configured adapter is the diagnostic `lora_step_0020.safetensors`; successful integration is not evidence of production-grade image quality.
- The SQLite database and uploaded/output files are local. Back up `data/` to preserve chat history. Do not delete `data/image_agent.sqlite3`, `data/uploads/`, or `data/outputs/` if you want to retain it.

## Setup / replace files

1. Stop the existing API and Streamlit processes (`Ctrl+C` in their terminals).
2. Back up the current project folder, especially `data/` and `.env`.
3. Extract this release and copy its project files over the existing project. Do **not** overwrite your existing `.env` or `data/` folder if you want to retain your private settings and generated files. This archive intentionally does not include secrets or runtime data.
4. If `.env` does not exist, copy `.env.example` to `.env` and configure the values below. If you already have a working `.env`, keep it; this archive does not contain secrets.
5. From the project directory:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Keep the existing virtual environment if it is already working; do not copy a virtual environment from another machine.

## Configuration

- `OLLAMA_BASE_URL` and `OLLAMA_MODEL` configure local Ollama.
- `GPU_BACKEND_URL` is the HTTPS ngrok base URL only, without `/v1/edit`.
- `GPU_BACKEND_TOKEN` must match the Kaggle `GPU_BACKEND_TOKEN` secret exactly.
- `APP_DATA_DIR` defaults to `data`.
- `OUTPUT_DIR` defaults to `data/outputs`.

Never commit or share `.env`. If a real token was exposed in an uploaded archive or terminal transcript, rotate it in Kaggle Secrets and update `.env`.

## Run locally

Terminal 1:

```bash
cd /path/to/image_agent
source .venv/bin/activate
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd /path/to/image_agent
source .venv/bin/activate
streamlit run app.py
```

Open the local Streamlit URL. Ollama should already be running; `ollama serve` reports “address already in use” when an Ollama server is already listening, which is usually fine.

## API routes

- `GET /health`
- `GET /v1/conversations`
- `POST /v1/conversations`
- `GET /v1/conversations/{id}`
- `PATCH /v1/conversations/{id}`
- `DELETE /v1/conversations/{id}`
- `POST /v1/conversations/{id}/messages` (multipart: `request`, optional `image`, `mask`, `reference`)
- `GET /v1/conversations/{id}/versions/{version}`
- `POST /v1/conversations/{id}/versions/{version}/restore`
- `GET /v1/files/{filename}`

The Kaggle GPU tunnel and notebook session must remain active during edits. The optional `gpu_service/server.py` is a separate local-GPU development service; the default application uses the Kaggle backend configured by `GPU_BACKEND_URL`. If the tunnel URL changes, update `GPU_BACKEND_URL` in `.env` and restart the local API.

## Automatic mask deployment update

Use the current `Part1_FINAL_DEPLOYMENT_AUTOMASK_COMPLETE.ipynb` Kaggle notebook with this local release. It adds Grounding DINO + SAM on CPU and makes the GPU API's `mask` field optional. After running the updated notebook, copy the HTTPS **base URL** (without `/v1/edit`) into `GPU_BACKEND_URL`. The local client automatically sends `auto_mask=true` and sends a supplied mask only when the user uploaded one. The local UI now describes this automatic-mask flow correctly. Restart local FastAPI and Streamlit after replacing files.

## Important: optional mask behavior in this fixed release

The local edit client does **not** reject a request when `mask` is absent. It sends the source image, planner-selected `target`, and `auto_mask=true`; if a manual mask exists it sends that mask and sets `auto_mask=false`. The Kaggle notebook endpoint then runs Grounding DINO + SAM when the mask is absent. If segmentation cannot locate the target, it returns an explicit error (HTTP 422) rather than a fake successful image.

For adding sunglasses, the target must be the existing placement region (for example `person's face`), not `sunglasses`. The reference image is saved in chat history but is not currently used for pixel-level model conditioning. The 20-step LoRA is diagnostic and is not quality-validated for production use.

### Safe replacement

Keep your existing local `.env` and `data/` directory when replacing code. This release intentionally excludes both the real `.env` and the local SQLite conversation database. Copy `.env.example` only if you need a template; do not overwrite your existing `.env`.
