# check_model_fixed.py
import os
import glob
import tensorflow as tf
import pickle
import numpy as np

print("🔍 Checking ACTUAL Model Status")
print("="*60)

# Find the latest model file
model_files = glob.glob("models/**/*.h5", recursive=True)
if model_files:
    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    print(f"📦 Latest model found: {latest_model}")
    
    try:
        # Load model
        model = tf.keras.models.load_model(latest_model)
        print("✅ Model loaded successfully!")
        
        # Model info
        print(f"\n📊 Model Architecture:")
        print(f"  Layers: {len(model.layers)}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Check weights
        total_params = model.count_params()
        print(f"  Total parameters: {total_params:,}")
        
        # Test prediction
        print(f"\n🧪 Testing prediction...")
        # Create dummy input matching your config (5000 features)
        dummy_input = np.random.randn(1, 5000).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        print(f"  Dummy prediction: {prediction[0][0]:.6f}")
        print(f"  Interpretation: {'SPAM' if prediction[0][0] > 0.5 else 'HAM'} (threshold=0.5)")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ No .h5 model files found!")

# Check vectorizer
print("\n" + "="*60)
print("🔤 Checking Vectorizer")
print("="*60)

vectorizer_files = glob.glob("models/**/*.pkl", recursive=True)
if vectorizer_files:
    # Find vectorizer
    for v_file in vectorizer_files:
        if "vectorizer" in v_file.lower() or "tfidf" in v_file.lower():
            print(f"📦 Vectorizer found: {v_file}")
            try:
                with open(v_file, 'rb') as f:
                    vectorizer = pickle.load(f)
                print("✅ Vectorizer loaded!")
                
                # Check vocabulary
                if hasattr(vectorizer, 'vocabulary_'):
                    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
                elif hasattr(vectorizer, 'get_feature_names_out'):
                    print(f"  Features: {len(vectorizer.get_feature_names_out())}")
                
                # Test transformation
                test_text = ["hello world test email"]
                try:
                    transformed = vectorizer.transform(test_text)
                    print(f"  Test transform shape: {transformed.shape}")
                except:
                    print("  Could not test transform")
                    
            except Exception as e:
                print(f"❌ Error loading vectorizer: {e}")
            break
else:
    print("❌ No vectorizer (.pkl) files found!")