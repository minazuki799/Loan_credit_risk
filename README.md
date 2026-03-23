# 🏦 Loan Credit Risk Predictor

A machine learning web app that predicts the probability of a loan applicant defaulting, built with XGBoost and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loancreditrisk-kpprxkmo2vyrmjumjukc9t.streamlit.app/)

---

## 📌 Project Overview
This app assists financial institutions in making faster, more consistent, 
and data-driven lending decisions by predicting loan default risk.

---

## 🤖 Model Performance
| Metric | Score |
|---|---|
| ROC AUC | 0.87 |
| Precision | 80% |
| Recall | 80% |
| F1 Score | 80% |
| Threshold | 0.31 |

---

## 🛠️ Tech Stack
- **Language:** Python
- **Model:** XGBoost + Scikit-learn
- **App:** Streamlit + Plotly
- **Data:** Pandas + NumPy
- **Custom Library:** swiftmltoolz

---

## 🚀 How to Run Locally

**Using pipenv:**
```bash
git clone https://github.com/minazuki799/Loan_credit_risk.git
cd Loan_credit_risk
pipenv install
pipenv run streamlit run streamlit_app.py
```

**Using pip:**
```bash
git clone https://github.com/minazuki799/Loan_credit_risk.git
cd Loan_credit_risk
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📁 Repository Structure
```
├── streamlit_app.py    # Streamlit web application
├── train.py            # Model training script
├── notebook.ipynb      # EDA and model development
├── credit_risk_model.pkl  # Saved model
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## 👤 Author
**Victor Okosun**
- GitHub: [minazuki799](https://github.com/minazuki799)
- LinkedIn: [victor-okosun](https://linkedin.com/in/victor-okosun)
```
