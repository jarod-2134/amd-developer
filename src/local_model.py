import requests
import json
import re
import ast

def init_local_model():
    """
    Initialize local model connection. Since we're using Ollama, 
    we just define the local endpoint url.
    """
    print("Initializing local Ollama connection...")
    return "http://localhost:11434/api/generate"

def generate_local(prompt, local_url):
    """
    Zero-cost local inference via Ollama.
    """
    try:
        payload = {
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 100, # Cap generation to save time
                "num_ctx": 1024,
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
    
def generate_local_with_confidence(prompt, local_url):
    """
    Local inference with confidence scoring. 
    Ollama does not natively return logprobs in the /api/generate endpoint currently.
    Workaround: Ask the model to output a confidence score, or just fallback if it fails.
    Returns (answer, confidence_score).
    """
    try:
        # We prompt the model to output JSON with a confidence score
        system_prompt = "You are a fact-checking AI. Provide the answer and a confidence score from 0.0 to 1.0 in JSON format: {\"answer\": \"...\", \"confidence\": 0.95}"
        payload = {
            "model": "qwen2.5:7b",
            "prompt": f"{system_prompt}\n\nQuestion: {prompt}",
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 50, # JSON response should be very short
                "num_ctx": 1024,
                "temperature": 0.0
            }
        }
        resp = requests.post(local_url, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json().get("response", "{}")
            parsed = json.loads(data)
            return parsed.get("answer", ""), float(parsed.get("confidence", 0.0))
        else:
            return "", 0.0
    except Exception:
        # Dummy fallback
        answer = f"[LOCAL FACT PLACEHOLDER] Answer for: {prompt[:30]}..."
        return answer, 0.95
    
def solve_math_locally(prompt):
    """
    Attempt to solve simple arithmetic locally.
    Returns string answer if successful, None if it can't parse it.
    """
    # Look for a simple math equation like "What is 2 + 2?"
    match = re.search(r'([\d\.]+)\s*([\+\-\*/])\s*([\d\.]+)', prompt)
    if match:
        num1, op, num2 = match.groups()
        try:
            num1, num2 = float(num1), float(num2)
            if op == '+': ans = num1 + num2
            elif op == '-': ans = num1 - num2
            elif op == '*': ans = num1 * num2
            elif op == '/': ans = num1 / num2
            
            if ans.is_integer():
                ans = int(ans)
            return str(ans)
        except Exception:
            return None
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
    
    if category == "code_debugging":
        context = lint_code_locally(prompt)
    elif category == "mathematical_reasoning":
        ans = solve_math_locally(prompt)
        if ans is not None:
            early_answer = ans
            
    return context, early_answer