from src.utils.config import get_config

config = get_config()
print(' Config loaded')
print(f'Project: {config.data_config["project"]["name"]}')
print(f'Version: {config.data_config["project"]["version"]}')
print(f'Data dir: {config.data_dir}')
print(f'Models dir: {config.models_dir}')
print(f'Results dir: {config.results_dir}')
print()
print(' Configuration OK! Vous pouvez lancer le Notebook 01')
