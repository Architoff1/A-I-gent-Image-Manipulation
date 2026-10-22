from pathlib import Path
from PIL import Image
import requests

from image.manipulator import execute_image_edit


class FakeResponse:
    headers = {"content-type": "image/png", "X-Mask-Mode": "automatic"}
    content = b"fake-png-bytes"
    ok = True
    status_code = 200
    text = ""

    def raise_for_status(self):
        return None


def test_missing_manual_mask_is_forwarded_as_auto_mask(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "white").save(source)
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("GPU_BACKEND_URL", "https://example.invalid")
    monkeypatch.setenv("GPU_BACKEND_TOKEN", "test-token")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    result = execute_image_edit({
        "current_image": str(source),
        "mask_image": None,
        "current_task": {
            "operation": "add",
            "description": "Add sunglasses",
            "target": "person's face",
            "constraints": ["preserve lighting"],
        },
        "image_history": [],
    })

    assert result.get("error") is None
    assert captured["url"] == "https://example.invalid/v1/edit"
    assert "mask" not in captured["files"]
    assert captured["data"]["auto_mask"] == "true"
    assert captured["data"]["target"] == "person's face"
    assert captured["headers"]["X-Backend-Token"] == "test-token"
    assert Path(result["current_image"]).is_file()
