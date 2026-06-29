# 📰 Fake News Detection using Machine Learning and NLP

## 📌 Project Overview

This project aims to detect whether a news article is **Fake** or **True** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The model is trained on news articles published between **August 2017 and December 2017**. The project performs text preprocessing, feature extraction using TF-IDF, trains multiple machine learning models, evaluates their performance, selects the best model based on evaluation metrics, and provides predictions through a Streamlit web application.

---

## 📂 Dataset

**Dataset Name:** ISOT Fake News Dataset

The dataset consists of two CSV files:

- `Fake.csv` – Fake news articles
- `True.csv` – Real news articles

### Dataset Description

- Time Period: **August 2017 – December 2017**
- Language: English
- Data Type: News Articles
- Classes:
  - Fake News (0)
  - True News (1)

---

## 📁 Project Structure

```text
Fake_News_Detection_ML/
│
├── app.py                     
├── README.md
├── requirements.txt
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── models_result/
│   ├── evaluation_results.csv
│   ├── classification_report.txt
│   └── confusion_matrix.txt
│
└── src/
    ├── data_loader.py
    ├── preprocess.py
    ├── feature_eng.py
    ├── model_training.py
    ├── evaluate.py
    └── predict.py
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

---

# Natural Language Processing (NLP)

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Tokenization
- Stopword removal
- Lemmatization

---

# Feature Engineering

The processed text is converted into numerical vectors using

**TF-IDF (Term Frequency-Inverse Document Frequency)**

The trained TF-IDF vectorizer is saved as

```
models/tfidf_vectorizer.pkl
```

---

# Machine Learning Models

The following classification algorithms are trained and evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest Classifier
4. Linear Support Vector Classifier (Linear SVC)

---

# Model Evaluation

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

The best model is selected using the following priority:

1. Highest F1 Score
2. Highest ROC-AUC Score (if F1 is tied)
3. Highest Accuracy (if both are tied)

---

# Saved Files

## Models

```
models/
```

- best_model.pkl
- tfidf_vectorizer.pkl

## Results

```
results/
```

- evaluation_results.csv
- classification_report.txt
- confusion_matrix.txt

---

# Running the Project

## Step 1

Clone the repository

```bash
git clone <repository-url>
```

---

## Step 2

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3

Train and evaluate the models

```bash
python src/evaluate.py
```

This will

- Train all models
- Evaluate performance
- Select the best model
- Save the trained model
- Save evaluation results

---

## Step 4

Run the Streamlit application

```bash
streamlit run app.py
```

---

# Prediction Workflow

```
Input News Article
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Best Trained Model
        │
        ▼
Prediction
        │
        ▼
Fake News / True News
```

---

# Example Output

```
Input:

Trump recognizes Jerusalem as Israel's capital.

Prediction:

✅ True News
```

```
Input:

NASA confirms Earth will experience six days of complete darkness.

Prediction:

🚨 Fake News
```

---

# Future Improvements

- Deep Learning (LSTM/Bi-LSTM)
- BERT-based Fake News Detection
- News URL Classification
- Explainable AI using SHAP/LIME
- Confidence Score Visualization
- Deployment on Streamlit Cloud

---

# Author

Developed as a Machine Learning and Natural Language Processing project using Python, Scikit-learn, and Streamlit.
