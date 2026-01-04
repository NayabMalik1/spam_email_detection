import yaml
from pathlib import Path

config_path = Path("configs/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("✓ Config loaded successfully")
print(f"API will run on: http://{config['api']['host']}:{config['api']['port']}")
print(f"Model: {config['model']['name']}")
print(f"Hidden layers: {config['model']['hidden_layers']}")