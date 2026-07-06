import os
import json
import sys
import openai
import re
import ast
import requests

# We will use Ollama via requests directly

def init_local_model():
    """
    Initialize local model connection. Since we're using Ollama, 
    we just define the local endpoint url.
    """
    print("Initializing local Ollama connection...")
    return "http://localhost:11434/api/generate"

def classify_task(prompt, local_url):
    """
    Placeholder to classify the task into one of the 8 categories.
    In reality, we would use the local model or a lightweight classifier here.
    """
    # Dummy logic for demonstration
    if "summarise" in prompt.lower():
        return "text_summarisation"
    elif "bug" in prompt.lower() or "fix" in prompt.lower():
        return "code_debugging"
    elif "+" in prompt or "calculate" in prompt.lower():
        return "mathematical_reasoning"
    elif "capital of" in prompt.lower() or "who is" in prompt.lower():
        return "factual_knowledge"
    return "unknown" # Default fallback

def generate_local(prompt, local_url):
    """
    Zero-cost local inference via Ollama.
    """
    try:
        # We assume the model 'llama3' (or phi3) is pulled in your Docker container
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(local_url, json=payload, timeout=10)
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
            "model": "llama3",
            "prompt": f"{system_prompt}\n\nQuestion: {prompt}",
            "stream": False,
            "format": "json"
        }
        resp = requests.post(local_url, json=payload, timeout=10)
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

def fetch_model_metadata(model_id):
    """
    Fetch model metadata locally if available, fallback to Hugging Face API dynamically.
    Returns tags and stop sequences.
    """
    tags = []
    stop_sequences = []
    
    # Attempt local lookup first
    local_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")
    try:
        if os.path.exists(local_db_path):
            with open(local_db_path, "r", encoding="utf-8") as f:
                models_db = json.load(f)
            
            # Find the model in the list
            model_data = next((m for m in models_db if m.get("id") == model_id), None)
            
            if model_data:
                tags = model_data.get("tags", [])
                config = model_data.get("config", {})
                tokenizer_config = config.get("tokenizer_config", {})
                
                if "eos_token" in tokenizer_config:
                    stop_sequences.append(tokenizer_config["eos_token"])
                    
                chat_template = config.get("chat_template_jinja", "")
                if "<|im_end|>" in chat_template:
                    stop_sequences.append("<|im_end|>")
                if "<turn|>" in chat_template:
                    stop_sequences.append("<turn|>")
                
                print(f"Loaded metadata for {model_id} from local models.json!")
                return tags, stop_sequences
    except Exception as e:
        print(f"Error reading local models.json: {e}")

    print(f"Model {model_id} not found locally, falling back to HTTP...")
    try:
        url = f"https://huggingface.co/api/models/{model_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            tags = data.get("tags", [])
            
            # Extract stop sequences
            config = data.get("config", {})
            tokenizer_config = config.get("tokenizer_config", {})
            
            if "eos_token" in tokenizer_config:
                stop_sequences.append(tokenizer_config["eos_token"])
                
            chat_template = config.get("chat_template_jinja", "")
            if "<|im_end|>" in chat_template:
                stop_sequences.append("<|im_end|>")
            if "<turn|>" in chat_template:
                stop_sequences.append("<turn|>")
                
    except Exception as e:
        print(f"Failed to fetch metadata for {model_id}: {e}")
        
    return tags, stop_sequences

def build_routing_table(models):
    """
    Build a routing dictionary dynamically based on HF tags.
    """
    routing = {
        "code_model": None,
        "video_model": None,
        "reasoning_model": None,
        "general_model": None,
        "stops": {}
    }
    
    for model in models:
        tags, stops = fetch_model_metadata(model)
        routing["stops"][model] = stops if stops else ["<eos>"] # Fallback stop
        
        # Determine strengths
        is_code = "coding" in tags or "custom_code" in tags or "code" in model.lower()
        is_video = "video" in tags or "multimodal" in tags
        is_reasoning = "reasoning" in tags or "math" in tags
        
        if is_code and not routing["code_model"]:
            routing["code_model"] = model
        if is_video and not routing["video_model"]:
            routing["video_model"] = model
        if is_reasoning and not routing["reasoning_model"]:
            routing["reasoning_model"] = model
            
        if not routing["general_model"]:
            routing["general_model"] = model
            
    return routing

def generate_remote(prompt, client, selected_model, category="unknown", stop_sequences=None):
    """
    Hit the Fireworks API for hard tasks, with optimized system prompts to save tokens.
    """
    system_prompts = {
        "mathematical_reasoning": "You are a math solver. Provide ONLY the final numerical answer. Do not show work.",
        "code_debugging": "Provide ONLY the corrected code. No explanations, no markdown formatting.",
        "code_generation": "Provide ONLY the generated code. No explanations, no markdown formatting.",
        "factual_knowledge": "Answer the factual question as concisely as possible, ideally in a single sentence or word."
    }
    system_instruction = system_prompts.get(category, "You are a helpful assistant. Provide extremely concise answers.")
    
    if client and selected_model and client.api_key != "dummy":
        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300,
                stop=stop_sequences if stop_sequences else ["<eos>"]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return f"[REMOTE ERROR] Failed to call API: {e}"
    else:
        # Fallback if no client (e.g. running locally without keys)
        return f"[REMOTE PLACEHOLDER] Answer for: {prompt[:30]}..."

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

    # 2. Read tasks from input file
    input_path = "/input/tasks.json"
    try:
        with open(input_path, "r") as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        print("Using dummy tasks for local testing...")
        tasks = [
            {"task_id": "t1", "prompt": "Summarise the following text in one sentence: The quick brown fox jumps over the lazy dog."},
            {"task_id": "t2", "prompt": "What is 2 + 2?"},
            {"task_id": "t3", "prompt": "Fix this bug: def foo(): retun 1"},
            {"task_id": "t4", "prompt": "What is the capital of France?"}
        ]

    # Initialize Local and Remote clients
    local_model = init_local_model()
    
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
    routing_table = build_routing_table(models)
    
    # 3. Process each task
    results = []
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        # Pass 1: Classify
        category = classify_task(prompt, local_model)
        
        # Dynamic Target Model Selection
        target_model = routing_table["general_model"] or selected_model
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
                answer = generate_remote(prompt, client, target_model, category, target_stops)
                
        elif category in ["code_debugging", "code_generation", "mathematical_reasoning"]:
            tool_context, early_answer = run_local_tools(prompt, category)
            if early_answer is not None:
                answer = early_answer
            else:
                augmented_prompt = prompt + tool_context
                answer = generate_remote(augmented_prompt, client, target_model, category, target_stops)
            
        else:
            # Default to remote for Logic, and unknown categories
            answer = generate_remote(prompt, client, target_model, category, target_stops)

        results.append({
            "task_id": task_id,
            "answer": answer
        })

    # 4. Write results to output file
    output_path = "/output/results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Successfully processed all tasks.")
    sys.exit(0)

if __name__ == "__main__":
    main()
