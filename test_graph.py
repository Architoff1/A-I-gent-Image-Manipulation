from agent.graph import graph


initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "Add three people near the building."
        }
    ],
    "current_image": "test_image.png",
    "reference_images": [],
    "image_history": [],
    "current_task": None,
    "route": None,
}


result = graph.invoke(initial_state)

print("\nFINAL STATE:")
print(result)
