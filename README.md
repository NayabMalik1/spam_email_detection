
# 📧 Email Spam Detection using Artificial Neural Network (ANN)

A complete **Machine Learning + Deep Learning** based Email Spam Detection system built using **Artificial Neural Networks (ANN)** and deployed using **FastAPI**.

This project classifies emails as **Spam** or **Not Spam** using trained ANN models and TF-IDF vectorization.

---

## 🚀 Features

- 🔍 Spam / Ham Email Classification
- 🧠 ANN (Artificial Neural Network) based model
- 📊 TF-IDF text vectorization
- ⚡ FastAPI backend
- 🌐 Web interface using HTML, CSS & JavaScript
- 📈 Model health & prediction statistics API
- 🧪 REST API support (JSON based predictions)

---

## 🏗️ Project Architecture

```

email_spam_ann/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/your_dataset
│   ├── processed/
│   └── dataset_metadata.json
│
├── models/
│   ├── __init__.py
│   ├── ann_model.py
│   ├── text_vectorizer.py
│   └── saved_models/
│       └── README.md
│
├── utils/
│   ├── __init__.py
│   ├── text_cleaner.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── logger.py
│
├── training/
│   ├── __init__.py
│   ├── train_ann.py
│   ├── optimizer_tuning.py
│   └── callbacks.py
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   └── evaluate.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── schemas.py
│   └── templates/
│       └── index.html
│
├── scripts/
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── test_model.py
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── logs/
├── outputs/
│   ├── plots/
│   └── reports/
└── temp/

````

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras**
- **FastAPI**
- **Scikit-learn**
- **NLTK**
- **Uvicorn**
- **HTML, CSS, JavaScript**

---

## 🧠 Model Details

- Model Type: **Artificial Neural Network (ANN)**
- Vectorization: **TF-IDF**
- Output:
  - `0` → Not Spam
  - `1` → Spam

---

## 🔧 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/NayabMalik1/spam_email_detection.git
cd spam_email_detection
````

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Resources

Run Python shell:

```bash
python
```

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

### 5️⃣ Model Files

Ensure the following files exist inside:

```
email_spam_ann/models/saved_models/
```

* `email_spam_classifier_ann_*.h5`
* `vectorizer_tfidf.pkl`
* `vectorizer_tfidf.json`

> ⚠️ If missing, retrain the model or place downloaded model files here.

---

## ▶️ Run the Application

```bash
python api/main.py
```

Server will start at:

```
http://127.0.0.1:8000
```

---

## 🌐 API Endpoints

| Endpoint       | Method | Description           |
| -------------- | ------ | --------------------- |
| `/`            | GET    | Web Interface         |
| `/api/predict` | POST   | Email spam prediction |
| `/api/health`  | GET    | Model & API health    |

---

## 🧪 Sample Prediction Request

```json
{
  "email": "Congratulations! You have won a free prize."
}
```

Response:

```json
{
  "prediction": "Spam",
  "confidence": 0.97
}
```

---

## 📌 Use Case

* Academic projects
* ANN / ML learning
* Email filtering systems
* AI-based text classification

---

## 👩‍💻 Author

**Nayab Zahoor**
Bachelor of Software Engineering
Email Spam Detection – ANN Project

---

## 📜 License

This project is for **educational purposes only**.
