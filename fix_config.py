import re

# Lire le fichier config.py
with open('src/utils/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Chercher et remplacer la propriété news_api_key
pattern = r'@property\s+def news_api_key\(self\):.*?return self\.data_config\[.*?\]'
replacement = '''@property
    def news_api_key(self):
        """Get News API key from environment."""
        return os.getenv("NEWS_API_KEY", "")'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Sauvegarder
with open('src/utils/config.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(' config.py modifié - news_api_key lit maintenant depuis .env')
