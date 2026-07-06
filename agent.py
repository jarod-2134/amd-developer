import os
import json
import sys

def main():
    # 1. Read environment variables injected by the harness
    try:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        models = os.environ["ALLOWED_MODELS"].split(",")
    except KeyError as e:
        print(f"Missing required environment variable: {e}")
        sys.exit(1)

    # Use the first allowed model as default, or select based on task
    selected_model = models[0] if models else None

    # 2. Read tasks from input file
    input_path = "/input/tasks.json"
    try:
        with open(input_path, "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        # Creating dummy tasks for local testing purposes if /input/ doesn't exist
        print("Using dummy tasks for local testing...")
        tasks = [
            {"task_id": "t1", "prompt": "Summarise the following text in one sentence: Hello world"},
            {"task_id": "t2", "prompt": "What is 2 + 2?"}
        ]

    # 3. Process each task
    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        # TODO: Initialize your LLM client here (e.g. using the openai package)
        # client = openai.OpenAI(
        #     api_key=api_key,
        #     base_url=base_url
        # )
        
        # TODO: Call the model to get the answer
        # response = client.chat.completions.create(
        #     model=selected_model,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # answer = response.choices[0].message.content
        
        # Placeholder for the generated answer
        answer = f"Placeholder answer for task {task_id}"

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    # 4. Write results to output file
    output_path = "/output/results.json"
    
    # Ensure output directory exists (useful for local testing)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Successfully processed all tasks.")
    sys.exit(0)

if __name__ == "__main__":
    main()
