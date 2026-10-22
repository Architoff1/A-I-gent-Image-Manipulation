"""Legacy placeholder for local ControlNet orchestration.

ControlNet conditioning for image edits is configured in the Kaggle GPU
deployment notebook. This local placeholder is not used by the active edit route.
"""
class ControlNetModel:
    def __init__(self):
        self.loaded = False

    def condition(self, image: str, control_type: str):
        raise NotImplementedError(
            "ControlNet conditioning runs in the Kaggle GPU backend, not this local placeholder."
        )
