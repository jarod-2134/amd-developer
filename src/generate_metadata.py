from dotenv import load_dotenv
from pathlib import Path

import os
import json
import requests

load_dotenv()  # Load environment variables from .env file

models = os.environ.get("ALLOWED_MODELS", "").split(",")

def _get_model_metadata(model_id):
    """
    Fetch metadata for a given model ID directly from the Fireworks REST API.
    Safely handles both full paths and short model slugs.
    """
    # Extract only the short model slug (e.g., 'minimax-m3') if a full path was passed
    model_slug = model_id.split("/")[-1]
    
    url = f"https://api.fireworks.ai/v1/accounts/fireworks/models/{model_slug}"
    headers = {
        "Authorization": f"Bearer {os.environ.get('FIREWORKS_API_KEY')}",
        "Accept": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Will raise an exception for 4xx/5xx errors
    return response.json()

def save_model_metadata_to_file(model_id, metadata):
    """
    Save the model metadata to a JSON file.
    """
    project_root = Path(__file__).resolve().parent.parent
    folder_path = project_root / "metadata"
    folder_path.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

    file_path = folder_path / f"{model_id}.json"
    json_string = json.dumps(metadata, indent=2)  # Convert metadata to JSON string with indentation

    file_path.write_text(json_string, encoding="utf-8")  # Write the JSON string to the file

def get_metadata_for_models():
    """
    Fetch and save metadata for all allowed models.
    """
    for model_id in models:
        metadata = _get_model_metadata(model_id)
        save_model_metadata_to_file(model_id, metadata)

def fetch_model_metadata_from_file(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
        
        model_id = model_data.get("name", file_path.stem)
        details = model_data.get("base_model_details", {}) or model_data.get("baseModelDetails", {})
        
        efficiency_metrics = {
            "parameter_count": None,
            "is_moe": details.get("moe", False),
            "supports_tools": model_data.get("supports_tools") or model_data.get("supportsTools", False),
            "context_length": model_data.get("context_length") or model_data.get("contextLength", 4096)
        }
        
        try:
            raw_params = details.get("parameter_count") or details.get("parameterCount")
            if raw_params:
                efficiency_metrics["parameter_count"] = int(raw_params)
        except ValueError:
            pass
            
        tags = []
        if efficiency_metrics["supports_tools"]: tags.append("tools")
        if efficiency_metrics["is_moe"]: tags.append("moe")
        if "coder" in model_id.lower() or "code" in model_id.lower(): tags.append("coding")

        conv_config = model_data.get("conversation_config", {}) or model_data.get("conversationConfig", {})
        chat_template = conv_config.get("template", "")
        all_text_configs = f"{chat_template} {model_data.get('description', '')}"
        
        stop_sequences = []
        if "<|im_end|>" in all_text_configs: stop_sequences.append("<|im_end|>")
        if "<turn|>" in all_text_configs: stop_sequences.append("<turn|>")
        if "<|eot_id|>" in all_text_configs: stop_sequences.append("<|eot_id|>")
        if not stop_sequences: stop_sequences.append("<eos>")

        return model_id, tags, stop_sequences, efficiency_metrics

    except Exception as e:
        print(f"Error parsing file metadata at {file_path}: {e}")
        return None