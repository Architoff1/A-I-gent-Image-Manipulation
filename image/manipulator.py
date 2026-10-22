"""Client for the authenticated Kaggle GPU image-editing backend."""
import os
import requests
from pathlib import Path
from agent.state import AgentState


def _build_inpaint_prompt(task: dict) -> str:
    operation = task.get("operation", "modify")
    description = (task.get("description") or "Edit the image as requested.").strip()
    target = (task.get("target") or "object").strip()
    if operation == "remove":
        instruction = f"Remove the {target} from the image and naturally reconstruct the background where it was."
    elif operation == "add":
        instruction = f"Add the requested element to the image, editing the {target} placement region."
    elif operation == "replace":
        instruction = f"Replace the {target} with the requested replacement."
    else:
        instruction = f"Modify the {target} as requested."
    # Keep the planner's description because it often contains the requested visual change.
    parts = [instruction]
    if description.lower() not in instruction.lower():
        parts.append(description)
    constraints = task.get("constraints") or []
    if constraints:
        parts.append("Preserve these requirements: " + "; ".join(str(x) for x in constraints))
    parts.append("Keep the result photorealistic and consistent with the original lighting, perspective, and texture.")
    return " ".join(parts)


def execute_image_edit(state: AgentState):
    task = state.get("current_task") or {}
    image_path = state.get("current_image")
    mask_path = state.get("mask_image")

    if not image_path or not Path(image_path).is_file():
        return {
            "response": "Please upload an image before requesting an edit.",
            "error": "missing_image",
        }
    operation = task.get("operation")
    if operation not in {"add", "remove", "modify", "replace"}:
        return {
            "response": f"Unsupported image-edit operation: {operation}",
            "error": "unsupported_operation",
        }

    has_manual_mask = bool(mask_path and Path(mask_path).is_file())
    base_url = os.getenv("GPU_BACKEND_URL", "").strip().rstrip("/")
    if not base_url:
        return {
            "response": "GPU_BACKEND_URL is not configured. Start the Kaggle deployment and set its HTTPS base URL in your local .env file.",
            "error": "missing_gpu_backend_url",
        }
    if base_url.endswith("/v1/edit"):
        base_url = base_url[:-len("/v1/edit")].rstrip("/")
    endpoint = f"{base_url}/v1/edit"
    backend_token = os.getenv("GPU_BACKEND_TOKEN", "").strip()
    if base_url.startswith("https://") and not backend_token:
        return {
            "response": "GPU_BACKEND_TOKEN is missing from the local environment. Configure the same token used by Kaggle Secrets.",
            "error": "missing_gpu_backend_token",
        }

    headers = {"X-Backend-Token": backend_token} if backend_token else {}
    full_prompt = _build_inpaint_prompt(task)
    timeout = max(30, min(int(os.getenv("GPU_BACKEND_TIMEOUT", "900")), 1800))

    try:
        with open(image_path, "rb") as image_file:
            files = {"image": (Path(image_path).name, image_file, "application/octet-stream")}
            mask_file = open(mask_path, "rb") if has_manual_mask else None
            try:
                if mask_file is not None:
                    files["mask"] = (Path(mask_path).name, mask_file, "application/octet-stream")
                response = requests.post(
                    endpoint,
                    headers=headers,
                    files=files,
                    data={
                        "prompt": full_prompt,
                        "operation": operation,
                        "target": task.get("target") or "object",
                        "auto_mask": "false" if has_manual_mask else "true",
                    },
                    timeout=(20, timeout),
                )
            finally:
                if mask_file is not None:
                    mask_file.close()

        if not response.ok:
            try:
                body = response.json()
                detail = body.get("detail") or body.get("error") or response.text
            except (ValueError, AttributeError):
                detail = response.text
            detail = str(detail).strip()[:700] or "No error details returned."
            return {
                "response": f"GPU backend returned HTTP {response.status_code}: {detail}",
                "error": f"gpu_http_{response.status_code}",
            }

        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            detail = response.text[:500] if hasattr(response, "text") else "No response body."
            return {
                "response": f"GPU backend returned an unexpected response type ({content_type or 'unknown'}). {detail}",
                "error": "gpu_unexpected_content_type",
            }

        project_root = Path(__file__).resolve().parents[1]
        configured_output = Path(os.getenv("OUTPUT_DIR", str(project_root / "data" / "outputs"))).expanduser()
        output_dir = configured_output if configured_output.is_absolute() else (project_root / configured_output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"edited_{os.urandom(8).hex()}.png"
        output_file.write_bytes(response.content)
        history = list(state.get("image_history", [])) + [str(image_path), str(output_file)]
        mask_mode = response.headers.get("X-Mask-Mode", "manual" if has_manual_mask else "automatic")
        return {
            "current_image": str(output_file),
            "image_history": history,
            "response": f"Image edit completed successfully (mask mode: {mask_mode}).",
            "error": None,
        }

    except requests.Timeout:
        return {
            "response": f"The GPU edit timed out after {timeout} seconds. Check that the Kaggle notebook and ngrok tunnel are still running; a cold model startup may take longer.",
            "error": "gpu_timeout",
        }
    except requests.RequestException as exc:
        return {
            "response": f"Could not reach the GPU backend. Check GPU_BACKEND_URL and confirm the Kaggle service and HTTPS tunnel are running. Details: {exc}",
            "error": "gpu_connection_error",
        }
    except OSError as exc:
        return {
            "response": f"Could not save the generated image locally: {exc}",
            "error": "output_write_error",
        }
