import pandas as pd
import os

print("Checking your dataset...")
print("="*60)

dataset_path = "data/raw/spam_Emails_data.csv"

if os.path.exists(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Records: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Check for text and label columns
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'message' in col.lower() or 'email' in col.lower()]
        label_cols = [col for col in df.columns if 'spam' in col.lower() or 'label' in col.lower() or 'target' in col.lower()]
        
        print(f"\nPossible text columns: {text_cols}")
        print(f"Possible label columns: {label_cols}")
        
        if label_cols:
            print(f"\nClass distribution:")
            print(df[label_cols[0]].value_counts())
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
else:
    print(f"❌ Dataset not found at: {dataset_path}")
    print("\nPlease make sure:")
    print(f"1. Your file is at: {dataset_path}")
    print("2. File name is exactly: spam_Emails_data.csv")
    print("3. File is in CSV format")

print("="*60)