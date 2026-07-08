import requests
import os
import json

from dotenv import load_dotenv

load_dotenv()

def get_all_deployments(account: str) -> list:
    api_key = os.environ.get('FIREWORKS_API_KEY', "")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key
    }

    response = requests.get(f"https://api.fireworks.ai/v1/{account}/deployments", headers=headers)
    response.raise_for_status()

    return response.json().get("deployments", [])

def deploy_model(account: str, model: str):
    api_key = os.environ.get('FIREWORKS_API_KEY', "")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key
    }

    payload = {
        "baseModel": model
    }

    response = requests.post(f"https://api.fireworks.ai/v1/{account}/deployments", headers=headers, json=payload)
    print(response.json())
    response.raise_for_status()

if __name__ == '__main__':
    deploy_model("accounts/kevinkruijthof", "accounts/fireworks/models/gemma-4-26b-a4b-it")