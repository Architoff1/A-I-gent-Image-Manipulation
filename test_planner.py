from agent.graph import graph

request = "Add three people near the building."

result = graph.invoke({
    "messages": [
        {"role": "user", "content": request}
    ],
    "current_image": None,
    "mask_image": None,
    "reference_images": [],
    "image_history": [],
    "current_task": None,
    "route": None,
})

task = result.get("current_task")
route = result.get("route")

print("Request:", request)
print("Task:", task)
print("Route:", route)
print("Response:", result.get("response"))
print("Error:", result.get("error"))

assert task is not None, "Planner did not produce a task."
assert task["intent"] == "image_edit", task
assert task["operation"] == "add", task
assert route == "image_edit", route

print("\nPASS: Planner classification and graph routing are correct.")
