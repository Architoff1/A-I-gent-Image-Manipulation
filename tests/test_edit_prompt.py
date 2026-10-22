from image.manipulator import _build_inpaint_prompt


def test_remove_prompt_explicitly_requests_background_reconstruction():
    prompt = _build_inpaint_prompt({
        "operation": "remove",
        "description": "Remove the bicycle",
        "target": "bicycle",
        "constraints": ["preserve shadows"],
    })
    assert "Remove the bicycle" in prompt
    assert "reconstruct the background" in prompt
    assert "preserve shadows" in prompt


def test_add_prompt_targets_existing_placement_region():
    prompt = _build_inpaint_prompt({
        "operation": "add",
        "description": "Add sunglasses",
        "target": "person's face",
        "constraints": [],
    })
    assert "person's face placement region" in prompt
