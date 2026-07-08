import requests
import os

from dotenv import load_dotenv

load_dotenv()

def _get_all_accounts() -> list:
    api_key = os.environ.get('FIREWORKS_API_KEY', "")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + api_key
    }

    response = requests.get("https://api.fireworks.ai/v1/accounts", headers=headers)
    response.raise_for_status()

    return response.json()


def get_account() -> dict:
    accounts = _get_all_accounts().get("accounts", [])

    return accounts[0]
