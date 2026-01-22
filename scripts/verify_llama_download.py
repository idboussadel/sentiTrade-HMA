"""
Vérification du téléchargement des adapters Llama-2
"""

from pathlib import Path
import sys

print("="*70)
print("VÉRIFICATION TÉLÉCHARGEMENT ADAPTERS LLAMA-2")
print("="*70)

# Chemin vers les adapters
adapter_path = Path("models/sentiment_llm/llama2_qlora_sentiment_final")

# Fichiers attendus avec tailles minimales (bytes)
expected_files = {
    "adapter_model.safetensors": 150_000_000,  # ~152 MB
    "adapter_config.json": 100,                 # ~1-2 KB
    "tokenizer.model": 400_000,                 # ~500 KB
    "tokenizer.json": 1_500_000,                # ~1.8 MB
    "tokenizer_config.json": 100,               # ~1 KB
    "special_tokens_map.json": 100,             # ~400 bytes
}

print(f"\n📂 Vérification du dossier : {adapter_path.absolute()}\n")

# Vérifier existence du dossier
if not adapter_path.exists():
    print("❌ ERREUR : Dossier non trouvé !")
    print(f"   Chemin : {adapter_path.absolute()}")
    print("\n💡 Assurez-vous d'avoir copié les fichiers dans :")
    print(f"   {adapter_path.absolute()}")
    sys.exit(1)

print("✅ Dossier trouvé\n")

# Vérifier chaque fichier
all_ok = True
total_size = 0

for filename, min_size in expected_files.items():
    file_path = adapter_path / filename
    
    if file_path.exists():
        size = file_path.stat().st_size
        size_mb = size / (1024 * 1024)
        total_size += size
        
        if size >= min_size:
            print(f"✅ {filename:40s} ({size_mb:7.2f} MB)")
        else:
            print(f"⚠️ {filename:40s} ({size_mb:7.2f} MB) - Trop petit !")
            all_ok = False
    else:
        print(f"❌ {filename:40s} (MANQUANT)")
        all_ok = False

# Résumé
print("\n" + "="*70)
if all_ok:
    print("✅ TOUS LES FICHIERS SONT PRÉSENTS ET VALIDES")
    print(f"\n📊 Taille totale : {total_size / (1024 * 1024):.2f} MB")
    print("\n🎯 Prochaine étape : Exécuter test_sentiment_integration.py")
else:
    print("❌ PROBLÈME DÉTECTÉ")
    print("\n💡 Actions à faire :")
    print("   1. Retélécharger les fichiers manquants depuis Google Drive")
    print("   2. Vérifier que le ZIP s'est bien décompressé")
    print("   3. S'assurer que les fichiers sont dans le bon dossier")

print("="*70 + "\n")
