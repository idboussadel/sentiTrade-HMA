import yaml
from pathlib import Path

config = {
    'sentiment_llm': {
        'model_name': 'ProsusAI/finbert',
        'model_type': 'transformer',
        'max_length': 512,
        'batch_size': 8
    },
    'tft': {
        'model_name': 'TFT',
        'architecture': 'temporal_fusion_transformer',
        'hyperparameters': {
            'hidden_size': 128,
            'lstm_layers': 2,
            'attention_heads': 4,
            'dropout': 0.1
        },
        'sequence_length': 30,
        'prediction_length': 5
    },
    'baseline_models': [
        {'name': 'ARIMA', 'params': {'p': 5, 'd': 1, 'q': 0}},
        {'name': 'LSTM', 'params': {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2}}
    ]
}

with open('configs/model_config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False)

print(' model_config.yaml créé')
