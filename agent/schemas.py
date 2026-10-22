from typing import Literal
from pydantic import BaseModel, Field, model_validator


class PlannerOutput(BaseModel):
    intent: Literal[
        "conversation",
        "image_generation",
        "image_edit",
        "image_analysis",
    ] = Field(description="Request type")

    operation: Literal[
        "none",
        "generate",
        "add",
        "remove",
        "modify",
        "replace",
        "analyze",
    ] = Field(description="Image operation")

    description: str = Field(description="Concise task description")
    target: str = Field(description="Target object or visual element")
    location: str = Field(default="", description="Where the operation applies")

    reference_required: bool = Field(
        default=False,
        description="Whether a reference image is required",
    )

    reference_indices: list[int] = Field(default_factory=list)

    constraints: list[str] = Field(
        default_factory=list,
        description="Requirements to preserve",
    )

    @model_validator(mode="after")
    def validate_intent_operation(self):
        valid_combinations = {
            "conversation": {"none"},
            "image_generation": {"generate"},
            "image_edit": {"add", "remove", "modify", "replace"},
            "image_analysis": {"analyze"},
        }

        if self.operation not in valid_combinations[self.intent]:
            raise ValueError(
                f"Invalid combination: intent={self.intent!r}, "
                f"operation={self.operation!r}"
            )

        return self
