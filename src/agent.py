import os
import json
import sys
import openai

from pathlib import Path

from local_model import build_routing_table, build_routing_table_deterministic, classify_task, init_local_model, generate_local, generate_local_with_confidence, run_local_tools, solve_math_locally
from remote_model import generate_remote
from generate_metadata import get_metadata_for_models

def main():
    # 1. Read environment variables injected by the harness
    try:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        models = os.environ["ALLOWED_MODELS"].split(",")
    except KeyError as e:
        print(f"Missing required environment variable: {e}")
        # Allow local bypass if we just want to run placeholders locally
        print("Bypassing for local testing...")
        api_key, base_url, models = "dummy", "dummy", ["dummy-model"]

    selected_model = models[0] if models else None
    project_root = Path(__file__).resolve().parent.parent
    total_tokens = 0

    # 2. Read tasks from input file
    input_path = project_root / "input" / "tasks.json"
    try:
        with open(input_path, "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        print("Using dummy tasks for local testing...")
        # tasks = [
        #     {"task_id": "t1", "prompt": "Summarise the following text in one sentence: The quick brown fox jumps over the lazy dog."},
        #     {"task_id": "t2", "prompt": "What is 2 + 2?"},
        #     {"task_id": "t3", "prompt": "Fix this bug: def foo(): retun 1"},
        #     {"task_id": "t4", "prompt": "What is the capital of France?"}
        # ]
        tasks = [
            {
                "task_id": "t1_hard", 
                "prompt": "Summarise the core macroeconomic conflict described here in exactly one sentence: The central bank aims to curb sticky core inflation by aggressively raising benchmark interest rates, which tightens credit conditions and suppresses consumer spending. However, the federal government is simultaneously rolling out a massive fiscal stimulus package for domestic infrastructure projects, which injects heavy liquidity back into the market and threatens to completely counteract the monetary tightening."
            },
            {
                "task_id": "t2_hard", 
                "prompt": "Solve this step-by-step: A cloud data pipeline processes 1200 records per minute using 2 parallel processing workers. If you scale up the infrastructure by adding 3 more identical workers, but the resulting network congestion drops the individual processing efficiency of *every* active worker by 15%, what is the new total throughput of the pipeline in records per minute?"
            },
            {
                "task_id": "t3_hard", 
                "prompt": "Identify the structural logical flaw in this Python code that causes index errors or skips elements during execution, and provide the corrected version:\n\ndef clean_inactive_users(users):\n    for i in range(len(users)):\n        if users[i]['status'] == 'inactive':\n            del users[i]\n    return users"
            },
            {
                "task_id": "t4_hard", 
                "prompt": "Which sovereign nation, when accounting for all of its global overseas territories and dependencies, possesses the highest number of distinct geographic time zones, and exactly how many does it hold?"
            },
            {
                "task_id": "t5_hard", 
                "prompt": "Extract all distinct Named Entities from this sentence, classifying them strictly into a clean list of PERSON, ORG, or LOCATION: 'Last Tuesday, Apple Martin took a flight to visit the Orange manufacturing facility in Orange County to finalize the cloud acquisition contract with asset managers representing Blackrock.'"
            },
            {
                "task_id": "t6_hard", 
                "prompt": "Generate a clean, optimized Python function utilizing a sliding window approach to determine the length of the longest substring without repeating characters. Include appropriate type hints and a clean docstring."
            },
            {
                "task_id": "t7_hard", 
                "prompt": "Deduce the solution based on these constraints: 1) Alex, Blake, and Casey each live in a different colored house: Red, Blue, or Green. 2) The resident of the Red house strictly drinks coffee. 3) Blake does not live in the Green house. 4) Casey lives directly adjacent to the Blue house and strictly drinks water. Who lives in the Red house?"
            },
            {
                "task_id": "t8_hard", 
                "prompt": "Classify the overall sentiment of this user review as Positive, Negative, or Neutral, and provide a single-sentence justification: 'I went into the theater fully expecting to despise this sci-fi reboot after the abysmal trailers. Amazingly, the witty dialogue and spectacular cinematography actually salvaged the second half, though the unearned cliffhanger ending still leaves an incredibly sour taste in my mouth.'"
            }
        ]

    # Initialize Local and Remote clients
    local_model = init_local_model()

    # Fetch all metadata
    get_metadata_for_models()
    
    # Initialize Fireworks API client
    client = None
    if api_key != "dummy":
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    else:
        print("Running in dummy mode. Fireworks API calls will be simulated.")

    # Build dynamic routing table
    print("Building dynamic routing table...")
    routing_table = build_routing_table_deterministic(models, require_serverless=True)

    print("Successfully build the routing table.")

    for k, v in routing_table.items():
        if k != "stops":
            print(f"{k} -> {v}")
    
    # 3. Process each task
    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        # Pass 1: Classify
        category = classify_task(prompt)
        
        # Dynamic Target Model Selection
        target_model = routing_table.get(category) if category != "" else selected_model
        print(f"Task {task_id}: Classified as '{category}'. Using model '{target_model}'.")
        if category in ["code_debugging", "code_generation"] and routing_table["code_model"]:
            target_model = routing_table["code_model"]
        elif category in ["logical_deductive_reasoning", "mathematical_reasoning"] and routing_table["reasoning_model"]:
            target_model = routing_table["reasoning_model"]
            
        target_stops = routing_table["stops"].get(target_model, [])
        
        # Pass 2: Execute based on category
        if category in ["sentiment_classification", "named_entity_recognition", "text_summarisation"]:
            answer = generate_local(prompt, local_model)
        
        elif category == "factual_knowledge":
            answer, confidence = generate_local_with_confidence(prompt, local_model)
            if confidence < 0.90:  # Strict confidence ceiling for facts
                response = generate_remote(prompt, client, target_model, category, target_stops)
                answer = response.get("content")
                print(f"Model: {target_model} used {response.get("total_tokens")} tokens.")
                total_tokens += response.get("total_tokens")
                
        elif category in ["code_debugging", "code_generation", "mathematical_reasoning"]:
            tool_context, early_answer = run_local_tools(prompt, category)
            if early_answer is not None:
                answer = early_answer
            else:
                augmented_prompt = prompt + tool_context
                response = generate_remote(augmented_prompt, client, target_model, category, target_stops)
                answer = response.get("content")
                print(f"Model: {target_model} used {response.get("total_tokens")} tokens.")
                total_tokens += response.get("total_tokens")
            
        else:
            # Default to remote for Logic, and unknown categories
            response = generate_remote(prompt, client, target_model, category, target_stops)
            answer = response.get("content")
            print(f"Model: {target_model} used {response.get("total_tokens")} tokens.")
            total_tokens += response.get("total_tokens")

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    # 4. Write results to output file
    output_folder = project_root / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Successfully processed all tasks.\nUsing {total_tokens} tokens in total.")
    sys.exit(0)

if __name__ == "__main__":
    main()
