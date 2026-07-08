import requests
import re
import ast
from difflib import SequenceMatcher

# Per-category bar for accepting the LOCAL answer as final. Categories where
# a wrong answer is a binary fail against the judge (math, logic, code) get a
# higher bar than categories the judge tends to grade more softly on
# (sentiment, summarisation) -- being wrong there is more forgiving.
CONFIDENCE_THRESHOLDS = {
    "factual_knowledge": 0.85,
    "mathematical_reasoning": 0.90,
    "sentiment_classification": 0.70,
    "text_summarisation": 0.70,
    "named_entity_recognition": 0.75,
    "code_debugging": 0.85,
    "code_generation": 0.85,
    "logical_deductive_reasoning": 0.95,  # effectively forces remote almost always
}

_ERROR_MARKERS = ("[LOCAL ERROR]", "[LOCAL PLACEHOLDER]", "[LOCAL FACT PLACEHOLDER]")

def _is_degenerate(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    return answer.strip().startswith(_ERROR_MARKERS)

def _structural_score(category: str, answer: str) -> float:
    """Cheap, deterministic shape check -- not correctness, just format sanity."""
    text = answer.strip()

    if category == "named_entity_recognition":
        return 1.0 if re.search(r'[\{\[]|\b(PERSON|ORG|LOCATION|DATE)\b', text, re.I) else 0.3

    if category == "text_summarisation":
        return 1.0 if len(re.findall(r'[.!?](?:\s|$)', text)) == 1 else 0.4

    if category == "sentiment_classification":
        return 1.0 if re.search(r'\b(positive|negative|neutral)\b', text, re.I) else 0.3

    if category in ("code_debugging", "code_generation"):
        code_match = re.search(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        code = code_match.group(1) if code_match else text
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.1
        except Exception:
            return 0.5

    if category == "mathematical_reasoning":
        return 1.0 if re.search(r'-?\d+(\.\d+)?', text) else 0.2

    if category == "logical_deductive_reasoning":
        return 0.2 if re.search(r'\b(maybe|possibly|could be|not sure)\b', text, re.I) else 0.8

    return 0.7 if len(text.split()) >= 3 else 0.3  # factual_knowledge / default

def _self_consistency_score(prompt, local_url, model, first_answer) -> float:
    """Free second opinion via resampling. Costs latency, not tokens."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 100, "num_ctx": 1024, "temperature": 0.4},
        }
        resp = requests.post(local_url, json=payload, timeout=15)
        if resp.status_code != 200:
            return 0.5  # inconclusive -- don't reward or penalize
        second_answer = resp.json().get("response", "").strip()
        if _is_degenerate(second_answer):
            return 0.3
        return SequenceMatcher(None, first_answer.lower(), second_answer.lower()).ratio()
    except requests.exceptions.RequestException:
        return 0.5

def estimate_local_confidence(prompt, category, local_answer, local_url, model) -> float:
    if _is_degenerate(local_answer):
        return 0.0

    structural = _structural_score(category, local_answer)
    consistency = _self_consistency_score(prompt, local_url, model, local_answer)

    # Format-strict categories weight shape heavier; open-ended reasoning
    # weights agreement-across-samples heavier since shape alone says little.
    if category in ("named_entity_recognition", "text_summarisation", "sentiment_classification"):
        confidence = 0.7 * structural + 0.3 * consistency
    else:
        confidence = 0.4 * structural + 0.6 * consistency

    return round(confidence, 3)

def should_use_local(category: str, confidence: float) -> bool:
    return confidence >= CONFIDENCE_THRESHOLDS.get(category, 0.85)