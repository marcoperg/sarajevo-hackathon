import json
from pathlib import Path

def get_user(name):
    try:
        with open(Path('user_data') / Path(f'{name}.json')) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

get_user('Marcoa')