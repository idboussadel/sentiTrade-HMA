#!/usr/bin/env python3
"""
Test de génération du modèle Llama-2
"""

import sys
sys.path.append('.')

from src.models.sentiment_analyzer import LlamaSentimentAnalyzer
import torch

def main():
    print("🔬 TEST DE GÉNÉRATION\n")
    
    # Charger modèle
    print("📥 Chargement du modèle...")
    analyzer = LlamaSentimentAnalyzer(
        adapter_path="models/sentiment_llm/llama2_qlora_sentiment_final"
    )
    print("✅ Chargé\n")
    
    # Test text
    text = "Apple stock surges 15% on record iPhone sales"
    prompt = analyzer.format_prompt(text)
    
    print("=" * 70)
    print("📝 PROMPT:")
    print("=" * 70)
    print(prompt)
    print()
    
    # Tokenize
    inputs = analyzer.tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(analyzer.model.device)
    
    print("=" * 70)
    print("🔢 TOKENS:")
    print("=" * 70)
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Input IDs: {inputs['input_ids'][0][:20]}...")  # First 20 tokens
    print()
    
    # Generate avec différentes configs
    configs = [
        {"max_new_tokens": 10, "do_sample": False, "name": "Greedy (default)"},
        {"max_new_tokens": 20, "do_sample": False, "name": "Greedy (20 tokens)"},
        {"max_new_tokens": 10, "do_sample": True, "temperature": 0.7, "name": "Sampling"},
    ]
    
    for config in configs:
        print("=" * 70)
        print(f"🎯 TEST: {config.pop('name')}")
        print("=" * 70)
        
        with torch.no_grad():
            outputs = analyzer.model.generate(
                **inputs,
                pad_token_id=analyzer.tokenizer.eos_token_id,
                eos_token_id=analyzer.tokenizer.eos_token_id,
                **config
            )
        
        # Decode
        full_response = analyzer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"📊 Output shape: {outputs.shape}")
        print(f"📊 Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
        print(f"\n📄 FULL RESPONSE:\n{full_response}\n")
        
        # Extract generated part
        if "[/INST]" in full_response:
            generated = full_response.split("[/INST]")[-1].strip()
            print(f"✂️  GENERATED PART: '{generated}'")
        else:
            print("⚠️  No [/INST] marker found")
        
        print()

if __name__ == "__main__":
    main()
