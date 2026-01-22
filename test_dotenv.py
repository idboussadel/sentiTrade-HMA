import os
from pathlib import Path
from dotenv import load_dotenv

# Afficher le chemin actuel
print(f'Working directory: {os.getcwd()}')

# Chercher .env
env_path = Path('.env')
print(f'.env existe: {env_path.exists()}')
print(f'.env chemin absolu: {env_path.absolute()}')

if env_path.exists():
    print(f'.env contenu:')
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(repr(content))

# Charger avec chemin explicite
load_dotenv('.env')
key = os.getenv('NEWS_API_KEY')
print(f'\nClé chargée: {key}')
print(f'Type: {type(key)}')
print(f'Longueur: {len(key) if key else 0}')
