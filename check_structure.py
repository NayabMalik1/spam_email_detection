import os

required_files = [
    "main.py",
    "requirements.txt",
    "configs/config.yaml",
    "data/raw/spam_Emails_data.csv",
    "utils/__init__.py",
    "utils/data_loader.py",
    "scripts/preprocess_data.py"
]

print("Checking project structure...")
print("="*40)

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - MISSING")

print("="*40)