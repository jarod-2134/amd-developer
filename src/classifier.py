import re
import json
import ollama

from typing import Dict, List, Tuple, Literal
from pydantic import BaseModel

class TaskClassifier(BaseModel):
    category: Literal[
        "factual_knowledge",
        "mathematical_reasoning",
        "sentiment_classification",
        "text_summarisation",
        "named_entity_recognition",
        "code_debugging",
        "logical_deductive_reasoning",
        "code_generation"
    ]

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "factual_knowledge": "Explaining concepts, definitions, and how things work.",
    "mathematical_reasoning": "Multi-step arithmetic, percentages, word problems, projections.",
    "sentiment_classification": "Labelling sentiment and justifying the classification.",
    "text_summarisation": "Condensing passages to a specific format or length constraint.",
    "named_entity_recognition": "Extracting and labelling entities (person, org, location, date).",
    "code_debugging": "Identifying bugs in code snippets and providing corrected implementations.",
    "logical_deductive_reasoning": "Constraint-based puzzles where all conditions must be satisfied.",
    "code_generation": "Writing correct, well-structured functions from a spec.",
}

CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    ("code_debugging", [
        r"\bbug\b", r"\bhas a bug\b", r"find (the )?bug", r"find and fix",
        r"fix (this|the) (function|code)", r"structural (logical )?flaw",
        r"corrected? (version|implementation)", r"index error",
    ]),
    ("code_generation", [
        r"\bwrite a (python )?function\b", r"\bimplement a function\b",
        r"\bgenerate a (clean|optimized|python)? ?function\b",
    ]),
    ("mathematical_reasoning", [
        r"\d+\s*%", r"\bhow many\b", r"\bhow much\b",
        r"solve (this )?step[- ]by[- ]step", r"\bpercent(age)?\b",
        r"\bthroughput\b", r"\bprojection\b",
    ]),
    ("logical_deductive_reasoning", [
        r"each (own|live)s?", r"different colou?red? house", r"\bwho (owns|lives)\b",
        r"\bdeduce\b", r"\bconstraint", r"strictly (drinks|lives|owns)",
    ]),
    ("sentiment_classification", [
        r"classify the (overall )?sentiment", r"sentiment of this review",
        r"positive,?\s*negative,?\s*(or\s*)?neutral",
    ]),
    ("text_summarisation", [
        r"\bsummar(i|z)e\b", r"\bin (exactly )?one sentence\b", r"\bcondense\b",
    ]),
    ("named_entity_recognition", [
        r"named entit", r"extract.*entit", r"\bPERSON\b.*\bORG\b", r"\bentities\b",
    ]),
    ("factual_knowledge", [
        r"what is the capital", r"which (sovereign )?(country|nation)",
        r"\bexplain\b", r"\bdefine\b", r"how (does|do) .* work",
    ]),
]

def classify_task_deterministic(task: str) -> str:
    """Zero-cost, zero-latency, zero-dependency classification pass."""
    text = task.lower()
    scores = {}
    for category, patterns in CATEGORY_PATTERNS:
        hits = sum(1 for p in patterns if re.search(p, text))
        if hits:
            scores[category] = hits
    return max(scores, key=scores.get) if scores else ""

def classify_task_llm(task: str, model: str) -> str:
    """LLM fallback -- only reached when the deterministic pass finds nothing."""
    categories_block = "\n".join(f"- {c}: {d}" for c, d in CATEGORY_DESCRIPTIONS.items())
    prompt = (
        "Classify the task below into exactly one category key. "
        "Respond with the key only, nothing else.\n\n"
        f"{categories_block}\n\nTask: {task}"
    )
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            format=TaskClassifier.model_json_schema(),
            options={"num_predict": 20, "num_ctx": 1024, "temperature": 0.0},
        )
        data = json.loads(response.get("response", "{}"))
        category = data.get("category", "")
        return category if category in CATEGORY_DESCRIPTIONS else ""
    except Exception as e:
        print(f"LLM classification fallback failed: {e}")
        return ""

def classify_task(task: str, model: str = "qwen2.5:3b-instruct") -> str:
    category = classify_task_deterministic(task)
    if category:
        return category
    print("No confident deterministic match — falling back to local LLM.")
    return classify_task_llm(task, model)