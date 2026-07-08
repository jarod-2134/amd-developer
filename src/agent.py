import os
import json
import sys
import openai

from pathlib import Path

from local_model import init_local_model, generate_local, generate_local_with_confidence, run_local_tools, LOCAL_MODEL_ID
from remote_model import generate_remote
from generate_metadata import get_metadata_for_models
from classifier import classify_task
from routing import build_routing_table_deterministic
from confidence import estimate_local_confidence, should_use_local

LOCAL_FIRST_CATEGORIES = {
    "sentiment_classification",
    "text_summarisation",
    "named_entity_recognition",
    "factual_knowledge",
    "code_debugging",
    "code_generation",
    "mathematical_reasoning",
}

def resolve_target(routing_table, category, selected_model):
    """Pull the remote model + stop sequences for a category, with a safe fallback."""
    entry = routing_table.get(category) if category else None
    target_model = entry.get("remote") if entry else selected_model
    target_stops = routing_table.get("stops", {}).get(target_model, [])
    return target_model, target_stops

def call_remote(prompt, client, target_model, category, target_stops):
    response = generate_remote(prompt, client, target_model, category, target_stops)
    tokens = response.get("total_tokens", 0)
    print(f"Model: {target_model} used {tokens} tokens.")
    return response.get("content"), tokens


def main():
    # 1. Read environment variables injected by the harness
    try:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        models = os.environ["ALLOWED_MODELS"].split(",")
    except KeyError as e:
        print(f"Missing required environment variable: {e}")
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
            print(f"{k} -> remote: {v.get("remote")}, local: {v.get("local")}")

    # 3. Process each task
    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt")

        category = classify_task(prompt)
        target_model, target_stops = resolve_target(routing_table, category, selected_model)
        print(f"Task {task_id}: Classified as '{category}'. Using model '{target_model}'.")

        answer = None
        tokens_used = 0

        early_answer = None
        tool_context = ""
        if category in ("code_debugging", "code_generation", "mathematical_reasoning"):
            tool_context, early_answer = run_local_tools(prompt, category)

        if early_answer is not None:
            answer = early_answer

        elif category in LOCAL_FIRST_CATEGORIES:
            local_answer = generate_local(prompt, local_model)
            confidence = estimate_local_confidence(
                prompt, category, local_answer, local_model, LOCAL_MODEL_ID
            )
            print(f"Task {task_id}: local confidence = {confidence}")

            if should_use_local(category, confidence):
                answer = local_answer
            else:
                augmented_prompt = prompt + tool_context if tool_context else prompt
                answer, tokens_used = call_remote(augmented_prompt, client, target_model, category, target_stops)

        else:
            answer, tokens_used = call_remote(prompt, client, target_model, category, target_stops)

        total_tokens += tokens_used
        results.append({"task_id": task_id, "answer": answer})

    # 4. Write results to output file
    output_folder = project_root / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Successfully processed all tasks.\nUsing {total_tokens} tokens in total.")
    sys.exit(0)

if __name__ == "__main__":
    main()
