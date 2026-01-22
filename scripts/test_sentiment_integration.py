"""
Test d'intégration du Llama-2 Sentiment Analyzer
Vérifie que le modèle se charge et fonctionne correctement
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("TEST INTÉGRATION LLAMA-2 SENTIMENT ANALYZER")
print("="*70)

# ===========================================
# ÉTAPE 1 : VÉRIFICATION DES DÉPENDANCES
# ===========================================

print("\n[1/5] Vérification des dépendances...")

required_packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'safetensors': 'SafeTensors',
}

missing = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} (MANQUANT)")
        missing.append(package)

if missing:
    print(f"\n❌ Packages manquants : {', '.join(missing)}")
    print("\n💡 Installation requise :")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)

print("   ✅ Toutes les dépendances sont installées")

# ===========================================
# ÉTAPE 2 : IMPORT DU MODULE
# ===========================================

print("\n[2/5] Import du module sentiment_analyzer...")

try:
    from src.models.sentiment_analyzer import LlamaSentimentAnalyzer
    print("   ✅ Module importé avec succès")
except ImportError as e:
    print(f"   ❌ Erreur d'import : {e}")
    print("\n💡 Vérifiez que :")
    print("   - Le fichier src/models/sentiment_analyzer.py existe")
    print("   - Le dossier src/models/ contient un fichier __init__.py")
    sys.exit(1)

# ===========================================
# ÉTAPE 3 : CHARGEMENT DU MODÈLE
# ===========================================

print("\n[3/5] Chargement du modèle...")
print("   ⏳ Cela peut prendre 30-60 secondes...")
print("   💡 Le modèle fait ~13 GB en mémoire (FP16)")

try:
    import warnings
    warnings.filterwarnings('ignore')
    
    adapter_path = "models/sentiment_llm/llama2_qlora_sentiment_final"
    
    analyzer = LlamaSentimentAnalyzer(
        adapter_path=adapter_path,
        device="auto",
    )
    
    print("\n   ✅ Modèle chargé avec succès !")
    
except Exception as e:
    print(f"\n   ❌ Erreur de chargement : {e}")
    print("\n💡 Solutions possibles :")
    print("   1. Vérifier que vous avez assez de RAM/VRAM (13+ GB)")
    print("   2. Si erreur CUDA, mettre device='cpu' (plus lent)")
    print("   3. Vérifier les logs ci-dessus pour plus de détails")
    sys.exit(1)

# ===========================================
# ÉTAPE 4 : TESTS UNITAIRES
# ===========================================

print("\n[4/5] Tests unitaires sur exemples financiers...")

test_cases = [
    {
        "text": "The company reported record earnings exceeding expectations by 25%.",
        "expected": "positive",
    },
    {
        "text": "Operating losses widened significantly during the quarter.",
        "expected": "negative",
    },
    {
        "text": "The company has 25 offices across Europe.",
        "expected": "neutral",
    },
    {
        "text": "Revenue surged driven by strong demand in Asia.",
        "expected": "positive",
    },
    {
        "text": "The firm announced major layoffs affecting 1000 employees.",
        "expected": "negative",
    },
]

print(f"\n   Exécution de {len(test_cases)} tests...\n")

results = []
for i, test in enumerate(test_cases, 1):
    try:
        predicted = analyzer.predict(test["text"])
        is_correct = predicted == test["expected"]
        
        results.append(is_correct)
        
        status = "✅" if is_correct else "❌"
        print(f"   {status} Test {i}/{len(test_cases)}")
        print(f"      Texte    : {test['text'][:55]}...")
        print(f"      Attendu  : {test['expected']}")
        print(f"      Prédit   : {predicted}")
        
        if not is_correct:
            print(f"      ⚠️ ERREUR DE PRÉDICTION")
        print()
        
    except Exception as e:
        print(f"   ❌ Test {i} échoué : {e}\n")
        results.append(False)

# Calcul accuracy
accuracy = sum(results) / len(results) * 100
print(f"   📊 Accuracy : {accuracy:.0f}% ({sum(results)}/{len(results)} correct)")

# ===========================================
# ÉTAPE 5 : TEST AVEC SCORES NUMÉRIQUES
# ===========================================

print("\n[5/5] Test des scores numériques...")

test_texts = [
    "Strong performance driven by record sales.",
    "The company faces mounting losses.",
    "The board consists of 12 members.",
]

print()
for text in test_texts:
    result = analyzer.analyze_with_scores(text)
    print(f"   Text      : {text}")
    print(f"   Sentiment : {result['sentiment']}")
    print(f"   Score     : {result['score']:+.1f}")
    print()

# ===========================================
# RÉSUMÉ FINAL
# ===========================================

print("="*70)
print("RÉSUMÉ DES TESTS")
print("="*70)

import torch

if torch.cuda.is_available():
    vram = torch.cuda.memory_allocated(0) / (1024**3)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n💾 GPU :")
    print(f"   - Device : {torch.cuda.get_device_name(0)}")
    print(f"   - VRAM   : {vram:.2f} GB / {vram_total:.0f} GB utilisés")
else:
    print(f"\n💾 CPU Mode (pas de GPU détecté)")

print(f"\n📊 Résultats :")
print(f"   - Tests réussis     : {sum(results)}/{len(results)}")
print(f"   - Accuracy          : {accuracy:.0f}%")
print(f"   - Modèle            : Llama-2-7B + LoRA")
print(f"   - Adapters fusionnés : 224 modules")

if accuracy >= 80:
    print(f"\n✅ SUCCÈS : Le modèle fonctionne correctement !")
    print(f"\n🎯 Prochaines étapes :")
    print(f"   1. Créer notebook 09_qlora_llama2_finetuning.ipynb")
    print(f"   2. Tester sur données réelles du projet")
    print(f"   3. Intégrer dans pipeline TFT")
elif accuracy >= 60:
    print(f"\n⚠️ ATTENTION : Performance moyenne ({accuracy:.0f}%)")
    print(f"   Le modèle fonctionne mais pourrait être amélioré")
else:
    print(f"\n❌ PROBLÈME : Performance faible ({accuracy:.0f}%)")
    print(f"   Vérifier que les adapters sont corrects")

print("\n" + "="*70 + "\n")
