import yaml
from pathlib import Path

config = {
    'llm_finetuning': {
        'epochs': 3,
        'batch_size': 8,
        'learning_rate': 0.0002,
        'optimizer': 'adamw'
    },
    'tft_training': {
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'early_stopping': {
            'enabled': True,
            'patience': 10
        }
    },
    'backtesting': {
        'initial_capital': 10000,
        'position_size': 0.1,
        'strategy': {
            'entry_threshold': 0.6,
            'exit_threshold': 0.4
        }
    },
    'hardware': {
        'device': 'cuda',
        'mixed_precision': True,
        'num_workers': 4
    }
}

with open('configs/training_config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False)

print(' training_config.yaml créé')
