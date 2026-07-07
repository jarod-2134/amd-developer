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
    folder_path = Path("metadata")
    folder_path.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

    file_path = folder_path / f"{model_id}.json"
    json_string = metadata.model_dump_json(indent=4)  # Convert metadata to JSON string with indentation

    file_path.write_text(json_string, encoding="utf-8")  # Write the JSON string to the file

def get_metadata_for_models():
    """
    Fetch and save metadata for all allowed models.
    """
    for model_id in models:
        metadata = _get_model_metadata(model_id)
        save_model_metadata_to_file(model_id, metadata)