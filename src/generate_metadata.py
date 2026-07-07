from dotenv import load_dotenv
from fireworks import Fireworks
from pathlib import Path
import os
import json

load_dotenv()  # Load environment variables from .env file

models = os.environ.get("ALLOWED_MODELS", "").split(",")

client = Fireworks(api_key=os.environ.get("FIREWORKS_API_KEY"), base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.com"))

def _get_model_metadata(model_id):
    """
    Fetch metadata for a given model ID from the Fireworks API.
    """
    response = client.models.get(model_id, account_id="fireworks")
    
    return response

def save_model_metadata_to_file(model_id, metadata):
    """
    Save the model metadata to a JSON file.
    """
    project_root = Path(__file__).resolve().parent.parent
    folder_path = project_root / "metadata"
    folder_path.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

    file_path = folder_path / f"{model_id}.json"
    json_string = metadata.model_dump_json(indent=2)  # Convert metadata to JSON string with indentation

    file_path.write_text(json_string, encoding="utf-8")  # Write the JSON string to the file

def _get_metadata_for_models():
    """
    Fetch and save metadata for all allowed models.
    """
    for model_id in models:
        metadata = _get_model_metadata(model_id)
        save_model_metadata_to_file(model_id, metadata)

def fetch_model_metadata(model_id):
    """
    Fetch model metadata locally from individual Fireworks metadata files.
    Extracts relevant information to determine efficiency and capabilities.
    """
    # Safe fallback values
    tags = []
    stop_sequences = []
    efficiency_metrics = {
        "parameter_count": None,
        "is_moe": False,
        "supports_tools": False,
        "supports_image": False,
        "context_length": 4096
    }

    # Normalize model id to match file names safely (replacing slashes if necessary)
    safe_filename = model_id.split("/")[-1]
    
    project_root = Path(__file__).resolve().parent.parent
    local_db_path = project_root / "metadata" / f"{safe_filename}.json"

    try:
        if local_db_path.exists():
            with open(local_db_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            
            # 1. Parse Capabilities & Optimization Metrics
            details = model_data.get("base_model_details", {}) or model_data.get("baseModelDetails", {})
            
            # String parsing parameter count securely
            try:
                raw_params = details.get("parameter_count") or details.get("parameterCount")
                if raw_params:
                    efficiency_metrics["parameter_count"] = int(raw_params)
            except ValueError:
                pass
                
            efficiency_metrics["is_moe"] = details.get("moe", False)
            efficiency_metrics["context_length"] = model_data.get("context_length") or model_data.get("contextLength", 4096)
            efficiency_metrics["supports_tools"] = model_data.get("supports_tools") or model_data.get("supportsTools", False)
            efficiency_metrics["supports_image"] = model_data.get("supports_image_input") or model_data.get("supportsImageInput", False)
            
            # Generate tags for routing filters
            if efficiency_metrics["supports_tools"]: tags.append("tools")
            if efficiency_metrics["supports_image"]: tags.append("multimodal")
            if efficiency_metrics["is_moe"]: tags.append("moe")

            # 2. Extract Stop Sequences from conversation configuration templates
            conv_config = model_data.get("conversation_config", {}) or model_data.get("conversationConfig", {})
            chat_template = conv_config.get("template", "")
            
            # General fallback check inside model description/text configs for safety
            all_text_configs = f"{chat_template} {model_data.get('description', '')}"
            if "<|im_end|>" in all_text_configs:
                stop_sequences.append("<|im_end|>")
            if "<turn|>" in all_text_configs:
                stop_sequences.append("<turn|>")
            if "<|eot_id|>" in all_text_configs:  # Standard llama3 end marker
                stop_sequences.append("<|eot_id|>")

            print(f"Loaded metadata for {model_id} successfully!")
            return tags, stop_sequences, efficiency_metrics

    except Exception as e:
        print(f"Error reading local metadata for {model_id}: {e}")
        
    return tags, stop_sequences, efficiency_metrics