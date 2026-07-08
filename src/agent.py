import os
import json
import sys
import openai

from pathlib import Path

from confidence import estimate_local_confidence, should_use_local
from local_model import (
    CATEGORIES,
    LOCAL_FIRST_CATEGORIES,
    LOCAL_MODEL_ID,
    build_routing_table,
    build_routing_table_deterministic,
    classify_task,
    init_local_model,
    run_local_prework,
)
from remote_model import generate_remote
from generate_metadata import get_metadata_for_models


def _project_root() -> Path:
    """
    Resolve the project root robustly.

    In the Docker container the app lives at /app/agent.py, so parent.parent
    would resolve to '/' (wrong). Prefer the current working directory when it
    looks like the app directory; otherwise fall back to the file's parent.
    """
    cwd = Path.cwd()
    # /app (container) or the repo root (local dev where cwd is the repo root)
    if (cwd / "metadata").exists() or (cwd / "src").exists() or (cwd / "agent.py").exists():
        return cwd
    return Path(__file__).resolve().parent.parent


def build_confirm_prompt(prompt, category, draft, context):
    """
    Build a tight 'confirm and refine' prompt for the remote model.

    Passing a local draft lets the remote model skip full re-derivation, which
    cuts completion tokens and latency while preserving accuracy (the remote
    model still verifies and corrects). The instruction is category-specific so
    the final output matches what the LLM-Judge expects.
    """
    confirm_instructions = {
        "Mathematical reasoning": "Verify the draft solution. Respond with ONLY the final numerical answer.",
        "Code debugging": "Verify the draft fix. Provide ONLY the final corrected code, no explanations.",
        "Code generation": "Verify and refine the draft. Provide ONLY the final code, no explanations.",
        "Factual knowledge": "Verify the draft answer. Answer as concisely as possible.",
        "Sentiment classification": "Verify the draft. Respond as 'LABEL: one-sentence justification'.",
        "Named entity recognition": "Verify the draft entities. List as 'TYPE: value', one per line. No prose.",
        "Text summarisation": "Verify the draft summary follows the task's length/format constraint. Provide ONLY the summary.",
        "Logical / deductive reasoning": "Verify the draft. Provide ONLY the final answer.",
    }
    instr = confirm_instructions.get(category, "Verify and refine. Provide a concise final answer.")

    parts = [f"{instr}", "", f"Task: {prompt}"]
    if draft:
        parts.append(f"Draft answer: {draft}")
    if context:
        parts.append(context.strip())
    parts.append("Final answer:")
    return "\n".join(parts)


def main():
    # 1. Read environment variables injected by the harness
    try:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    except KeyError as e:
        print(f"Missing required environment variable: {e}")
        print("Bypassing for local testing...")
        api_key, base_url, models = "dummy", "dummy", ["dummy-model"]

    selected_model = models[0] if models else None
    project_root = _project_root()
    total_tokens = 0

    # 2. Read tasks from input file
    input_path = project_root / "input" / "tasks.json"
    try:
        with open(input_path, "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        print("Using dummy tasks for local testing...")
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

    # Fetch all metadata (resilient: bundled metadata/ is a fallback)
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
    routing_table = build_routing_table(models, require_serverless=True)

    # Guard: if no serverless models passed the filter (or metadata was missing),
    # fall back to routing every category to the first allowed model so the run
    # never crashes on a missing key.
    if not routing_table:
        print("⚠️ Routing table is empty — falling back to a single-model table.")
        fallback_model = selected_model or "dummy-model"
        routing_table = {cat: fallback_model for cat in CATEGORIES}
        routing_table["stops"] = {}

    print("Successfully built the routing table.")
    for k, v in routing_table.items():
        if k != "stops":
            # print(f"{k} -> remote: {v.get("remote")}, local: {v.get("local")}")
            print(f"{k} -> {v}")

    # 3. Process each task
    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt")

        # Pass 1: Classify (category + difficulty) — local, zero tokens
        category, difficulty = classify_task(prompt)

        # Resolve the target model + stop sequences for this category.
        if category in routing_table:
            target_model = routing_table[category]
        else:
            # Unknown / empty category → use the first allowed model.
            target_model = selected_model
        target_stops = routing_table.get("stops", {}).get(target_model, [])

        print(f"Task {task_id}: Classified as '{category}' (difficulty={difficulty}). "
              f"Using model '{target_model}'.")

        # Pass 2: Local prework (draft + deterministic context + confidence).
        # This runs for EVERY task — local tokens are free per Track 1 rules.
        draft, context, confidence = run_local_prework(prompt, category, difficulty, local_model)
        # confidence = estimate_local_confidence(prompt, category, draft, local_model, LOCAL_MODEL_ID)

        answer = None

        # --- Local-only gate: answer for free when easy + confident ----------
        # Only for capability domains where a local answer can be judge-accurate,
        # and only when the model is confident. Otherwise we escalate.
        if (
            category in LOCAL_FIRST_CATEGORIES
            and (difficulty == "easy" or difficulty == "medium")
            and should_use_local(category, confidence)
            and draft
        ):
            answer = draft
            print(f"Task {task_id}: answered locally (confidence={confidence}). 0 tokens.")
        else:
            print(f"Task {task_id}: answering remote (confidence={confidence}).")

        # --- Prework + confirm: every other case goes to remote --------------
        # The local draft is passed to the remote model to verify/refine, which
        # minimizes the remote completion length (and thus tokens).
        if answer is None:
            confirm_prompt = build_confirm_prompt(prompt, category, draft, context)
            response = generate_remote(confirm_prompt, client, target_model, category, target_stops)
            answer = response.get("content")
            tok = response.get("total_tokens", 0)
            print(f"Task {task_id}: remote confirm used {tok} tokens.")
            total_tokens += tok

        # Safety net: if everything failed, fall back to the local draft (even
        # if low-confidence) so we always emit an answer rather than empty.
        if not answer:
            answer = draft or "[NO ANSWER]"

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
