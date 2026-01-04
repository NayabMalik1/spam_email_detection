import sys
print(f"Python Version: {sys.version}")
print(f"Virtual Environment: {sys.prefix}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow Version: {tf.__version__}")
    print(f"✅ TensorFlow GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError as e:
    print(f"❌ TensorFlow Error: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas Version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas Error: {e}")

try:
    import sklearn
    print(f"✅ scikit-learn Version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn Error: {e}")

try:
    import nltk
    print("✅ NLTK Installed")
except ImportError as e:
    print(f"❌ NLTK Error: {e}")

try:
    import fastapi
    print(f"✅ FastAPI Version: {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI Error: {e}")

print("\n" + "="*50)
print("Installation Verification Complete!")
print("="*50)