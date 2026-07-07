import requests
import json
import re
import ast
import ollama

from typing import Literal, List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from pathlib import Path
from ollama import AsyncClient

from generate_metadata import fetch_model_metadata_from_file

# ---------------------------------------------------------------------------
# Local model configuration
# ---------------------------------------------------------------------------
# Single source of truth for the local Ollama model. The Dockerfile must pull
# this same model. If the built image exceeds the 10GB compressed limit, change
# this constant (and the matching `ollama pull` line in the Dockerfile) to
# "qwen2.5:7b" — no other code changes are required.
LOCAL_MODEL = "qwen3:14b"

# The eight capability categories defined by Track 1 of the hackathon.
CATEGORIES = [
    "Factual knowledge",
    "Mathematical reasoning",
    "Sentiment classification",
    "Text summarisation",
    "Named entity recognition",
    "Code debugging",
    "Logical / deductive reasoning",
    "Code generation",
]

# Categories whose answers can be produced locally for free when the model is
# confident. These map to Track 1's "easy" capability domains.
LOCAL_FIRST_CATEGORIES = {
    "Sentiment classification",
    "Named entity recognition",
    "Text summarisation",
    "Factual knowledge",
}


# ---------------------------------------------------------------------------
# Classification schema
# ---------------------------------------------------------------------------
class TaskClassifier(BaseModel):
    category: Literal[
        "Factual knowledge",          # Explaining concepts, definitions, and how things work
        "Mathematical reasoning",     # Multi-step arithmetic, percentages, word problems, projections
        "Sentiment classification",   # Labelling sentiment and justifying the classification
        "Text summarisation",         # Condensing passages to a specific format or length constraint
        "Named entity recognition",   # Extracting and labelling entities (person, org, location, date)
        "Code debugging",             # Identifying bugs in code snippets and providing corrected implementations
        "Logical / deductive reasoning",  # Constraint-based puzzles where all conditions must be satisfied
        "Code generation"             # Writing correct, well-structured functions from a spec
    ]
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description=(
            "easy: a single trivial step, short input, obvious answer. "
            "medium: a couple of steps or moderate-length input. "
            "hard: multi-step reasoning, long input, or complex constraints."
        )
    )


class DynamicRoutingTable(BaseModel):
    factual_knowledge: str = Field(
        alias="Factual knowledge",
        description="Model for explaining concepts, definitions, and how things work."
    )
    mathematical_reasoning: str = Field(
        alias="Mathematical reasoning",
        description="Model for multi-step arithmetic, percentages, word problems, projections."
    )
    sentiment_classification: str = Field(
        alias="Sentiment classification",
        description="Model for labelling sentiment and justifying the classification."
    )
    text_summarisation: str = Field(
        alias="Text summarisation",
        description="Model for condensing passages to a specific format or length constraint."
    )
    named_entity_recognition: str = Field(
        alias="Named entity recognition",
        description="Model for extracting and labelling entities (person, org, location, date)."
    )
    code_debugging: str = Field(
        alias="Code debugging",
        description="Model for identifying bugs in code snippets and providing corrected implementations."
    )
    logical_deductive_reasoning: str = Field(
        alias="Logical / deductive reasoning",
        description="Model for constraint-based puzzles where all conditions must be satisfied."
    )
    code_generation: str = Field(
        alias="Code generation",
        description="Model for writing correct, well-structured functions from a spec."
    )
    stops: Dict[str, List[str]] = Field(
        description="A dictionary mapping every analyzed model name exactly to its list of valid stop sequences."
    )

    # Ensures compatibility whether accessing via dictionary keys or attributes
    model_config = {"populate_by_name": True, "populate_by_alias": True}


def init_local_model():
    """
    Initialize local model connection. Since we're using Ollama,
    we just define the local endpoint url.
    """
    print("Initializing local Ollama connection...")
    return "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# Local generation primitives
# ---------------------------------------------------------------------------
def generate_local(prompt, local_url):
    """
    Zero-cost local inference via Ollama.
    """
    try:
        payload = {
            "model": LOCAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 100,  # Cap generation to save time
                "num_ctx": 2048,
                "temperature": 0.0
            }
        }
        resp = requests.post(local_url, json=payload, timeout=25)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            return f"[LOCAL ERROR] {resp.status_code}"
    except requests.exceptions.RequestException:
        # Fallback if Ollama isn't running yet during dev
        return f"[LOCAL PLACEHOLDER] Answer for: {prompt[:30]}..."


def generate_local_with_confidence(prompt, local_url, instruction=None):
    """
    Local inference with confidence scoring.
    Ollama does not natively return logprobs in the /api/generate endpoint currently.
    Workaround: Ask the model to output a confidence score, or just fallback if it fails.
    Returns (answer, confidence_score).

    An optional `instruction` can shape the answer format per category.
    """
    system_prompt = (
        instruction
        or "You are a fact-checking AI. Provide the answer and a confidence score "
           "from 0.0 to 1.0 in JSON format: {\"answer\": \"...\", \"confidence\": 0.95}"
    )
    try:
        payload = {
            "model": LOCAL_MODEL,
            "prompt": f"{system_prompt}\n\nQuestion: {prompt}",
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 150,  # JSON response should be short
                "num_ctx": 2048,
                "temperature": 0.0
            }
        }
        resp = requests.post(local_url, json=payload, timeout=35)
        if resp.status_code == 200:
            data = resp.json().get("response", "{}")
            parsed = json.loads(data)
            return parsed.get("answer", "").strip(), float(parsed.get("confidence", 0.0))
        else:
            return "", 0.0
    except Exception:
        # Dummy fallback: low confidence so the caller escalates to remote.
        # (Previous 0.95 passed the gate and silently returned placeholders.)
        return "", 0.0


# ---------------------------------------------------------------------------
# Deterministic local tools (prework)
# ---------------------------------------------------------------------------
def _solve_arithmetic_chain(expr):
    """Safely evaluate an arithmetic-only expression (digits and + - * / ( ) .)."""
    # Strip everything that isn't a digit, operator, paren, dot or space.
    cleaned = re.sub(r'[^\d\+\-\*/\(\)\.\s]', '', expr).strip()
    if not cleaned or not re.search(r'\d', cleaned):
        return None
    # Must contain at least one operator to be an "expression".
    if not re.search(r'[\+\-\*/]', cleaned):
        return None
    try:
        ans = eval(cleaned, {"__builtins__": {}}, {})
        if isinstance(ans, (int, float)) and not isinstance(ans, bool):
            if isinstance(ans, float) and ans.is_integer():
                ans = int(ans)
            return str(ans)
    except Exception:
        return None
    return None


def solve_math_locally(prompt):
    """
    Attempt to solve arithmetic locally. Handles single binary operations
    ("What is 2 + 2?") and chained arithmetic ("1200 * 5 * 0.85").
    Returns a string answer if fully solved, None otherwise.
    """
    # Look for sequences of digits/operators/parens that form an expression.
    for match in re.finditer(r'([\d][\d\.\s\+\-\*/\(\)]*[\d\)])', prompt):
        candidate = match.group(1)
        # Avoid grabbing a single bare number.
        if not re.search(r'[\+\-\*/]', candidate):
            continue
        ans = _solve_arithmetic_chain(candidate)
        if ans is not None:
            return ans
    return None


def lint_code_locally(prompt):
    """
    Attempt to extract and parse python code locally.
    Returns a syntax error context string, or empty string.
    """
    code_match = re.search(r'```(?:python)?\n(.*?)\n```', prompt, re.DOTALL)
    code_to_check = code_match.group(1) if code_match else prompt

    try:
        ast.parse(code_to_check)
        return ""
    except SyntaxError as e:
        return f"\n[LOCAL PRE-PROCESSING] Linter found a SyntaxError on line {e.lineno}: {e.msg}. Please fix this."
    except Exception:
        return ""


def run_local_tools(prompt, category):
    """
    Run local deterministic tools. Returns (context_string, early_answer).
    early_answer is set if the tool fully solved the task.
    """
    context = ""
    early_answer = None

    if category == "Code debugging":
        context = lint_code_locally(prompt)
    elif category == "Mathematical reasoning":
        ans = solve_math_locally(prompt)
        if ans is not None:
            early_answer = ans

    return context, early_answer


# ---------------------------------------------------------------------------
# Classification (uses the official Ollama SDK with structured output)
# ---------------------------------------------------------------------------
_CLASSIFY_INSTRUCTION = """You are a task router. Classify the user's task into exactly one category and rate its difficulty.

Categories:
- "Factual knowledge": explaining concepts, definitions, how things work
- "Mathematical reasoning": multi-step arithmetic, percentages, word problems, projections
- "Sentiment classification": labelling sentiment and justifying the classification
- "Text summarisation": condensing passages to a specific format or length constraint
- "Named entity recognition": extracting and labelling entities (person, org, location, date)
- "Code debugging": identifying bugs in code snippets and providing corrected implementations
- "Logical / deductive reasoning": constraint-based puzzles where all conditions must be satisfied
- "Code generation": writing correct, well-structured functions from a spec

Difficulty:
- "easy": a single trivial step, short input, obvious answer
- "medium": a couple of steps or moderate-length input
- "hard": multi-step reasoning, long input, or complex constraints
"""


def classify_task(task: str) -> Tuple[str, str]:
    """
    Classify a task into (category, difficulty) using the Ollama SDK with
    structured-output schema enforcement. Returns ("", "") on failure.
    """
    try:
        response = ollama.generate(
            model=LOCAL_MODEL,
            prompt=f"{_CLASSIFY_INSTRUCTION}\n\nTask: {task}",
            # Pass the Pydantic schema directly into the format argument so the
            # model is constrained to one of the 8 categories + a difficulty.
            format=TaskClassifier.model_json_schema(),
            options={
                "num_predict": 50,
                "num_ctx": 2048,
                "temperature": 0.0
            }
        )

        # The SDK returns the raw response text; parse it ourselves.
        data = json.loads(response.get("response", "{}"))
        return data.get("category", ""), data.get("difficulty", "")
    except Exception as e:
        print(f"Failed to classify task: {e}")
        return "", ""


# ---------------------------------------------------------------------------
# Unified local prework
# ---------------------------------------------------------------------------
# Per-category local instructions that shape the answer format so it matches
# what the LLM-Judge expects, and so the draft is directly usable.
_LOCAL_INSTRUCTIONS = {
    "Sentiment classification": (
        "Classify the sentiment as Positive, Negative, or Neutral, then give a "
        "single-sentence justification. Return JSON: "
        '{"answer": "<label>: <one sentence>", "confidence": 0.9}'
    ),
    "Named entity recognition": (
        "Extract all distinct named entities and classify each as PERSON, ORG, "
        "or LOCATION. Return JSON: "
        '{"answer": "TYPE: value\\nTYPE: value\\n...", "confidence": 0.9}'
    ),
    "Text summarisation": (
        "Summarise the passage following any length/format constraint in the task. "
        'Return JSON: {"answer": "<summary>", "confidence": 0.9}'
    ),
    "Factual knowledge": (
        "Answer the factual question as concisely as possible. Return JSON: "
        '{"answer": "<answer>", "confidence": 0.9}'
    ),
    "Mathematical reasoning": (
        "Solve the problem step by step, then return the final numerical answer. "
        'Return JSON: {"answer": "<final number>", "confidence": 0.9}'
    ),
    "Code debugging": (
        "Identify the bug and provide the corrected code only. Return JSON: "
        '{"answer": "<corrected code>", "confidence": 0.9}'
    ),
    "Code generation": (
        "Write the requested function with type hints and a docstring. Return JSON: "
        '{"answer": "<code>", "confidence": 0.9}'
    ),
    "Logical / deductive reasoning": (
        "Reason through the constraints and give the final answer. Return JSON: "
        '{"answer": "<final answer>", "confidence": 0.9}'
    ),
}


def run_local_prework(prompt: str, category: str, difficulty: str, local_url: str) -> Tuple[str, str, float]:
    """
    Unified local prework across all categories.

    Returns (draft, context, confidence):
      - draft: a candidate answer produced locally (may be empty)
      - context: deterministic prework context to append to a remote prompt
                 (e.g. a lint hint). Empty string when none.
      - confidence: the local model's self-reported confidence (0.0 if none)

    For math, a deterministic solver is tried first and short-circuits with
    confidence 1.0 when it fully solves the expression.
    """
    context = ""
    draft = ""
    confidence = 0.0

    # --- Deterministic fast paths first ---
    if category == "Mathematical reasoning":
        ans = solve_math_locally(prompt)
        if ans is not None:
            return ans, "", 1.0

    if category == "Code debugging":
        context = lint_code_locally(prompt)

    # --- Local-model draft for every category ---
    instruction = _LOCAL_INSTRUCTIONS.get(category)
    if instruction:
        draft, confidence = generate_local_with_confidence(
            prompt, local_url, instruction=instruction
        )

    return draft, context, confidence


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------
def fetch_single_profile(metadata_folder: Path, model_id: str) -> tuple:
    """
    Directly targets and reads the specific JSON file based on the model ID slug.
    """
    # Extract slug (e.g., "accounts/fireworks/models/gemma-4-26b-a4b-it" -> "gemma-4-26b-a4b-it")
    model_slug = model_id.split("/")[-1]
    json_file = metadata_folder / f"{model_slug}.json"

    print(f"Fetching profile for model: {model_slug}")

    if not json_file.exists():
        print(f"Warning: Expected file metadata/{model_slug}.json does not exist. Skipping.")
        return None

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        # Use the full name from the file if available, otherwise fallback to the requested ID
        resolved_id = model_data.get("name") or model_id
        details = model_data.get("base_model_details", {}) or model_data.get("baseModelDetails", {}) or {}

        efficiency_metrics = {
            "parameter_count": None,
            "is_moe": details.get("moe", False) or details.get("moe", False),
            "supports_tools": model_data.get("supports_tools") or model_data.get("supportsTools", False),
            "context_length": model_data.get("context_length") or model_data.get("contextLength", 4096),
            "supports_serverless": model_data.get("supports_serverless") or model_data.get("supportsServerless", False)
        }

        try:
            raw_params = details.get("parameter_count") or details.get("parameterCount")
            if raw_params:
                efficiency_metrics["parameter_count"] = int(raw_params)
        except (ValueError, TypeError):
            pass

        tags = []
        if efficiency_metrics["supports_tools"]: tags.append("tools")
        if efficiency_metrics["is_moe"]: tags.append("moe")
        if "code" in resolved_id.lower() or "coder" in resolved_id.lower(): tags.append("coding")

        conv_config = model_data.get("conversation_config", {}) or model_data.get("conversationConfig", {}) or {}
        chat_template = conv_config.get("template", "")
        all_text_configs = f"{chat_template} {model_data.get('description', '')}"

        stop_sequences = []
        if "<|im_end|>" in all_text_configs: stop_sequences.append("<|im_end|>")
        if "<turn|>" in all_text_configs: stop_sequences.append("<turn|>")
        if "<|eot_id|>" in all_text_configs: stop_sequences.append("<|eot_id|>")
        if not stop_sequences: stop_sequences.append("<eos>")

        return resolved_id, tags, stop_sequences, efficiency_metrics
    except Exception as e:
        print(f"Error reading metadata file {json_file.name}: {e}")
        return None


def build_routing_table(model_ids: List[str], require_serverless: bool = True, local_url="http://localhost:11434/api/generate"):
    project_root = Path(__file__).resolve().parent.parent
    metadata_folder = project_root / "metadata"

    compiled_metadata = {}
    stops_map = {}

    for m_id in model_ids:
        parsed = fetch_single_profile(metadata_folder, m_id)
        if parsed:
            resolved_id, tags, stop_tokens, metrics = parsed

            # --- SERVERLESS FILTER ---
            if require_serverless and not metrics.get("supports_serverless", False):
                print(f"⚠️ Skipping {resolved_id}: Requires On-Demand deployment (Serverless unsupported).")
                continue

            stops_map[resolved_id] = stop_tokens
            compiled_metadata[resolved_id] = {
                "inferred_tags": tags,
                "efficiency_metrics": metrics
            }

    if not compiled_metadata:
        print("Error: No models available matching the specified serverless requirement constraints.")
        return None

    prompt = f"""
    You are an infrastructure optimization engine. Allocate available models directly across specific task categories to maximize TOKEN EFFICIENCY.
    Speed and latency do not matter. Efficiency means matching tasks to the lowest computational/parameter footprint capable of maintaining perfect execution accuracy.

    ALLOWED TARGETS (You must choose strictly from these exact keys for the slots):
    {list(compiled_metadata.keys())}

    TARGET TASK CATEGORIES (Assign one of the allowed target model strings directly to each of these json keys):
    - "Sentiment classification"
    - "Text summarisation"
    - "Named entity recognition"
    - "Factual knowledge"
    - "Code generation"
    - "Code debugging"
    - "Mathematical reasoning"
    - "Logical / deductive reasoning"

    AVAILABLE INFRASTRUCTURE DATA:
    {json.dumps(compiled_metadata, indent=2)}

    STOPS TOKENS MAP (Pass this object exactly into your return payload 'stops' key):
    {json.dumps(stops_map)}

    TOKEN-EFFICIENCY ASSIGNMENT RULES:
    - "Sentiment classification" & "Text summarisation": Assign the model ID with the lowest absolute parameter count.
    - "Named entity recognition": Select the lowest parameter model ID that has 'supports_tools': true.
    - "Factual knowledge": Choose the model ID balancing factual capacity and active token overhead (Prioritize models where 'is_moe' is true).
    - "Code generation" & "Code debugging": Select the model ID containing 'coding' tags or explicit programming nomenclature.
    - "Mathematical reasoning" & "Logical / deductive reasoning": Map your heaviest, highest parameter, or deepest reasoning model ID here to guarantee complex tasks don't waste tokens on execution failures.
    """

    try:
        payload = {
            "model": LOCAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": DynamicRoutingTable.model_json_schema(),
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192
            }
        }

        response = requests.post(local_url, json=payload, timeout=180)
        if response.status_code == 200:
            return json.loads(response.json().get("response", "{}"))
        else:
            print(f"Ollama worker rejected request: {response.status_code}")
            return None
    except Exception as e:
        print(f"Critical execution error during token routing optimization: {e}")
        return None


def build_routing_table_deterministic(model_ids: List[str], require_serverless: bool = True):
    project_root = Path(__file__).resolve().parent.parent
    metadata_folder = project_root / "metadata"

    profiles = []
    for m_id in model_ids:
        parsed = fetch_single_profile(metadata_folder, m_id)
        if parsed:
            resolved_id, tags, stop_tokens, metrics = parsed

            # --- SERVERLESS FILTER ---
            if require_serverless and not metrics.get("supports_serverless", False):
                print(f"⚠️ Skipping {resolved_id}: Requires On-Demand deployment (Serverless unsupported).")
                continue

            profiles.append({
                "model_id": resolved_id,
                "parameter_count": metrics.get("parameter_count") or 0,
                "is_moe": metrics.get("is_moe", False),
                "supports_tools": metrics.get("supports_tools", False),
                "is_coding": "coding" in tags or "code" in resolved_id.lower(),
                "stop_sequences": stop_tokens
            })

    if not profiles:
        print("Error: No models available matching the specified serverless requirement constraints.")
        return None

    # Determine core optimal engine baselines
    low_intensity = min(profiles, key=lambda x: x["parameter_count"])["model_id"]
    high_compute = max(profiles, key=lambda x: x["parameter_count"])["model_id"]

    tool_capable = [p for p in profiles if p["supports_tools"]]
    structured_token = min(tool_capable, key=lambda x: x["parameter_count"])["model_id"] if tool_capable else low_intensity

    coding_models = [p for p in profiles if p["is_coding"]]
    domain_specialized = coding_models[0]["model_id"] if coding_models else high_compute

    moe_models = [p for p in profiles if p["is_moe"]]
    if moe_models:
        parametric_knowledge = moe_models[0]["model_id"]
    else:
        sorted_by_size = sorted(profiles, key=lambda x: x["parameter_count"])
        parametric_knowledge = sorted_by_size[len(sorted_by_size) // 2]["model_id"]

    # Final map matching the TaskClassifier categories directly
    routing_table = {
        "Sentiment classification": low_intensity,
        "Text summarisation": low_intensity,
        "Named entity recognition": structured_token,
        "Factual knowledge": parametric_knowledge,
        "Code generation": domain_specialized,
        "Code debugging": domain_specialized,
        "Mathematical reasoning": high_compute,
        "Logical / deductive reasoning": high_compute,
        "stops": {p["model_id"]: p["stop_sequences"] for p in profiles}
    }

    return routing_table
