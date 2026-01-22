#!/usr/bin/env python3
"""
Démo live pour la soutenance - SentiTrade-HMA
Date: December 31, 2024
"""

import sys
sys.path.append('.')

from src.models.sentiment_analyzer import LlamaSentimentAnalyzer
import time

def main():
    print("=" * 70)
    print(" " * 15 + "🚀 SENTITR ADE-HMA - DÉMO LIVE")
    print("=" * 70)
    print()
    
    # Charger modèle
    print("📥 Chargement du modèle Llama-2 + QLoRA...")
    start = time.time()
    
    analyzer = LlamaSentimentAnalyzer(
        adapter_path="models/sentiment_llm/llama2_qlora_sentiment_final"
    )
    
    load_time = time.time() - start
    print(f"\n✅ Modèle chargé en {load_time:.1f} secondes\n")
    
    # Exemples pré-chargés
    examples = [
        "Apple stock surges 15% on record iPhone sales",
        "Tesla faces production delays amid chip shortage",
        "Microsoft announces disappointing quarterly earnings",
        "The company reported strong revenue growth",
        "Operating losses widened significantly this quarter"
    ]
    
    print("=" * 70)
    print("🎯 EXEMPLES DE TESTS")
    print("=" * 70)
    
    # Analyser les exemples
    for i, text in enumerate(examples, 1):
        print(f"\n[{i}] 📰 {text}")
        print("    ⏳ Analyse en cours...")
        
        start = time.time()
        sentiment = analyzer.predict(text)
        score = analyzer.get_sentiment_score(sentiment)
        inference_time = time.time() - start
        
        # Emojis et couleurs
        emoji = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "⚪"
        
        print(f"    {emoji} Sentiment : {sentiment.upper()}")
        print(f"    📊 Score     : {score:+.1f}")
        print(f"    ⏱️  Temps     : {inference_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("🎤 MODE INTERACTIF")
    print("=" * 70)
    print("\nEntrez une phrase financière (ou 'quit' pour quitter) :")
    print("💡 Tapez 1-5 pour utiliser les exemples ci-dessus\n")
    
    # Mode interactif
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir !")
                break
            
            # Utiliser exemple si numéro
            if user_input.isdigit() and 1 <= int(user_input) <= len(examples):
                text = examples[int(user_input) - 1]
                print(f"\n📰 Texte : {text}")
            elif user_input:
                text = user_input
            else:
                continue
            
            # Analyse
            print("⏳ Analyse en cours...")
            start = time.time()
            sentiment = analyzer.predict(text)
            score = analyzer.get_sentiment_score(sentiment)
            inference_time = time.time() - start
            
            # Affichage
            emoji = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "⚪"
            
            print(f"\n{emoji} Sentiment : {sentiment.upper()}")
            print(f"📊 Score     : {score:+.1f}")
            print(f"⏱️  Temps     : {inference_time:.2f}s")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}\n")

if __name__ == "__main__":
    main()
