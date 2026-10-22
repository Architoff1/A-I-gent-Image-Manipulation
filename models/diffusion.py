"""Legacy placeholder; active image edits run on the Kaggle GPU backend.

The deployed SDXL Inpainting + ControlNet + LoRA pipeline is initialized by the
Kaggle deployment notebook and invoked through ``image.manipulator``. This
class is not used by the active LangGraph edit route.
"""
class DiffusionModel:
    def __init__(self):
        self.loaded = False

    def generate(self, prompt: str):
        raise NotImplementedError(
            "Text-to-image generation is not exposed by the current Kaggle edit endpoint."
        )

    def edit(self, image: str, prompt: str, mask: str | None = None):
        raise NotImplementedError(
            "Use the authenticated Kaggle /v1/edit endpoint through image.manipulator."
        )
