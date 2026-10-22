from agent.state import AgentState
from agent.llm import planner_llm, llm
from image.manipulator import execute_image_edit
from agent.schemas import PlannerOutput

PLANNER_PROMPT = """
You are a task planner for a general-purpose image manipulation agent.

Classify the user's latest request using exactly one intent and one operation.

INTENTS AND OPERATIONS:

1. conversation / none
   Use for ordinary conversation unrelated to image generation,
   image inspection, or image editing.

2. image_generation / generate
   Use when the user wants a new image created from a prompt.

3. image_edit / add
   Use when the user wants objects or visual elements added
   to an existing image.

4. image_edit / remove
   Use when the user wants objects or visual elements removed
   from an existing image.

5. image_edit / modify
   Use when the user wants an existing object or its appearance
   changed, such as changing a car's colour.

6. image_edit / replace
   Use when the user wants an existing object replaced
   with another object.

7. image_analysis / analyze
   Use when the user wants an existing image inspected or described
   without modifying it.

MANDATORY RULES:

- Never combine conversation with an image-editing operation.
- Never classify an explicit request to add, remove, modify, or
  replace something in an existing image as ordinary conversation.
- A request to add people or objects to an existing image is image_edit/add.
- A request to create a completely new image is image_generation/generate.
- Keep the description concise and actionable.
- For remove/modify/replace, set target to the existing object that should be segmented.
- For add, set target to the existing host/placement region that should be edited (e.g. "person's face" when adding sunglasses, "wall" when adding a picture). Do not set target to a new object that is not present in the source image.
- Record preservation requirements in constraints.
- Set reference_required to true only when supplied reference images
  are essential to completing the request.
- Return structured output only.
"""

def planner_node(state: AgentState):
    messages = state.get("messages", [])

    latest = next(
        (
            m.get("content", "")
            for m in reversed(messages)
            if m.get("role") == "user"
        ),
        "",
    )

    if not latest.strip():
        return {
            "error": "No user request was supplied.",
            "current_task": None,
        }

    try:
        result = planner_llm.invoke(
            f"{PLANNER_PROMPT}\n\nUSER REQUEST:\n{latest}"
        )

        task = (
            result.model_dump()
            if hasattr(result, "model_dump")
            else result.dict()
        )

        intent = task.get("intent")
        operation = task.get("operation")

        # Deterministic placement-region corrections for common additions. The new
        # object itself is not present yet, so segmentation must target its host region.
        request_lower = latest.lower()
        if operation == "add":
            placement_rules = [
                (("sunglasses", "eyeglasses", "glasses", "spectacles"), "person's face"),
                (("hat", "cap", "beanie"), "person's head"),
                (("necklace", "chain"), "person's neck and upper chest"),
                (("earrings", "earring"), "person's ears"),
                (("picture", "painting", "photo frame", "wall art"), "wall"),
            ]
            for terms, placement in placement_rules:
                if any(term in request_lower for term in terms):
                    task["target"] = placement
                    break

        # Repair the intent when the model's operation is unambiguous.
        operation_to_intent = {
            "none": "conversation",
            "generate": "image_generation",
            "add": "image_edit",
            "remove": "image_edit",
            "modify": "image_edit",
            "replace": "image_edit",
            "analyze": "image_analysis",
        }

        expected_intent = operation_to_intent.get(operation)

        if expected_intent and intent != expected_intent:
            task["intent"] = expected_intent

        # Validate the final result before routing.
        validated = PlannerOutput.model_validate(task)

        return {
            "current_task": validated.model_dump(),
            "error": None,
        }

    except Exception as exc:
        return {
            "error": (
                "Planner failed to produce a valid task. "
                f"Details: {exc}"
            ),
            "current_task": None,
        }

def router_node(state: AgentState):
    if state.get("error") or not state.get("current_task"):
        return {"route": "conversation"}
    intent = state["current_task"].get("intent")
    routes = {"conversation":"conversation", "image_generation":"image_generation", "image_edit":"image_edit", "image_analysis":"image_analysis"}
    return {"route": routes.get(intent, "conversation")}

def conversation_node(state: AgentState):
    if state.get("error"):
        return {"response": state["error"]}
    latest = next((m.get("content", "") for m in reversed(state.get("messages", [])) if m.get("role") == "user"), "")
    try:
        answer = llm.invoke(latest)
        return {"response": getattr(answer, "content", str(answer)), "error": None}
    except Exception as exc:
        return {"response": f"The local language model could not respond. Check Ollama. Details: {exc}", "error": str(exc)}

def image_generation_node(state: AgentState):
    return {"response": "Image generation is not connected yet. The current GPU service supports masked image editing; generation-only requests need a separate generation endpoint."}

def image_edit_node(state: AgentState):
    return execute_image_edit(state)

def image_analysis_node(state: AgentState):
    return {"response": "Image analysis is not connected yet. Upload an image and request an edit for the currently implemented workflow."}
