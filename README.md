# ai_textbased_fake_content_detector
# AI Fake Content Detector

A beginner-friendly project to classify text as either **AI-generated** or **Human-written** using basic NLP and machine learning. Runs entirely on free tools like Google Colab, NLTK, and scikit-learn.

---

## 🚀 Project Objective

Build a machine learning model that can predict whether a given sentence or paragraph is written by an AI (like ChatGPT) or a human.

---

## 📌 Problem Statement

With the rise of advanced language models, it's becoming harder to distinguish between human-written and AI-generated text. This project aims to detect fake/synthetic content using lightweight NLP techniques.

---

## 📂 Dataset

* **Filename:** `ai_human_text_dataset.csv`
* **Entries:** 150 labeled examples

  * 75 AI-generated
  * 75 Human-written
* Format: CSV with two columns:

  * `text`: the input sentence
  * `label`: either `AI` or `Human`

---

## 🧰 Tools & Technologies

* Python
* Google Colab (Free)
* Pandas
* NLTK (Natural Language Toolkit)
* Scikit-learn (ML models)

---

## 📊 Workflow

### 1. Data Preprocessing

* Lowercasing
* Tokenization
* Stopword removal
* Feature extraction with CountVectorizer or TfidfVectorizer

### 2. Model Building

* Train/Test Split (e.g., 80/20)
* Classifiers used:

  * Naive Bayes or
  * Logistic Regression

### 3. Evaluation

* Accuracy
* Confusion Matrix
* Predicting custom sentences

---

## 🔍 Sample Code (Google Colab)

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("ai_human_text_dataset.csv")

# Convert labels to binary
df["label"] = df["label"].apply(lambda x: 1 if x == "AI" else 0)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📌 Example Usage

```python
# Predict a new sentence
sentence = ["This tool analyzes data patterns using machine learning"]
sentence_vec = vectorizer.transform(sentence)
print("AI" if model.predict(sentence_vec)[0] == 1 else "Human")
```

---

## 🧠 Future Improvements

* Use deep learning models (e.g., LSTM or BERT)
* Add grammatical feature extraction (POS tagging, syntax trees)
* Deploy as a web app

---

## 📃 License

MIT License

---

## 🙌 Acknowledgments

Inspired by real-world problems in education and journalism caused by AI-generated fake content.

---

## 📞 Contact

Made with ❤️ by PavanSai Pothala

email: [pavansaipothala123@gmail.com](mailto:pavansaipothala123@gmail.com)
