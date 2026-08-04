# 🚢 Titanic Survival Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Deployment](https://img.shields.io/badge/Render-Ready-brightgreen.svg)](https://render.com/)

An end-to-end AI/Machine Learning web application that predicts passenger survival probability on the Titanic using a pre-trained **Scikit-Learn Logistic Regression Pipeline**, served via a **Flask REST API**, and presented through a modern, responsive **Glassmorphism SaaS Frontend**.

---

## ✨ Current Project Status

- ✅ **Machine Learning Pipeline**: Trained Scikit-Learn pipeline using `StandardScaler` feature normalization, 5-Fold `Cross Validation`, and `GridSearchCV` hyperparameter tuning (Final Test Accuracy: **79.33%**).
- ✅ **Flask Backend**: Production-ready REST API (`POST /predict`) with robust parameter validation, automatic `FamilySize` calculation (`SibSp + Parch + 1`), and one-hot encoding for `Embarked` ports.
- ✅ **Interactive Glassmorphism Frontend**: Navy Blue & Crisp White palette with frosted glass cards, numerical percentage counter animation, dynamic probability progress bar, and instant parameter feedback.
- ✅ **Cloud Deployment Ready**: Fully configured for **Render Cloud** deployment with `render.yaml` Blueprint, `gunicorn` WSGI server, and `requirements.txt`.

---

## 🛠️ Tech Stack

- **Backend & ML**: Python, Flask, Scikit-Learn, Pandas, NumPy, Joblib, Gunicorn
- **Frontend**: HTML5, CSS3 (Glassmorphism design system), Vanilla JavaScript, Font Awesome 6, Google Fonts (`Outfit` & `Inter`)
- **Data & Modeling**: Kaggle Titanic Dataset, Jupyter Notebook (`notebooks/Titanic_Survival_Predictor.ipynb`)
- **Deployment**: Render Cloud (`render.yaml`)

---

## 📁 Directory Structure

```
Titanic-Survival-Predictor/
├── app.py                      # Flask REST API & Web Server
├── titanic_model.pkl           # Trained Scikit-Learn Model Pipeline
├── requirements.txt            # Production Python Dependencies
├── render.yaml                 # Render Blueprint Deployment Configuration
├── README.md                   # Project Documentation
├── .gitignore                  # Git Exclusion Rules
│
├── templates/
│   └── index.html              # HTML5 Glassmorphism UI (Jinja2)
│
├── static/
│   ├── style.css               # Modern CSS System & Animations
│   └── script.js               # Async Frontend JS & API Fetch Handler
│
├── notebooks/
│   └── Titanic_Survival_Predictor.ipynb   # Model Training Notebook
│
├── data/
│   └── train.csv               # Kaggle Titanic Manifest Data
│
└── screenshots/                # Application UI Screenshots
```

---

## 📊 Model Information

| Specification | Details |
| :--- | :--- |
| **Dataset** | Kaggle Titanic Dataset (`data/train.csv`) |
| **Model Algorithm** | `Logistic Regression` (`C=0.01`) |
| **Preprocessing** | `StandardScaler` |
| **Validation** | 5-Fold Cross Validation |
| **Hyperparameter Tuning** | `GridSearchCV` |
| **Final Test Accuracy** | `79.33%` |
| **Features Used** | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `Embarked_Q`, `Embarked_S` |

---

## 🔌 API Endpoint Specification

### `POST /predict`

**Headers**: `Content-Type: application/json`

#### Request Payload:
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29.0,
  "Fare": 85.0,
  "SibSp": 0,
  "Parch": 0,
  "Embarked": "S"
}
```

#### JSON Response:
```json
{
  "prediction": "Survived",
  "prediction_value": 1,
  "probability": 75.11
}
```

---

## 🚀 Local Setup & Running

### 1. Clone the Repository
```bash
git clone https://github.com/XCodeBunnyX/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Flask Server
```bash
python3 app.py
```
Open **http://localhost:5001** (or `http://localhost:5000`) in your web browser.

---

## ☁️ Deployment on Render

This repository includes a `render.yaml` Blueprint for 1-click deployment on Render:

1. Log into your **[Render Dashboard](https://dashboard.render.com/)**.
2. Select **New +** → **Blueprint**.
3. Connect your GitHub repository (`XCodeBunnyX/Titanic-Survival-Prediction`).
4. Render will automatically build (`pip install -r requirements.txt`) and start the application using `gunicorn app:app`.

---

## 👤 Author

Built by **Bunny** (CodeXBunny AI Team)
