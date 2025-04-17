# CreditCardFraud_Detection

# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using real-world data. The project uses advanced models, smart data handling, and feature engineering techniques to build an effective fraud detection system.

---

## 📌 Overview

Credit card fraud is a major financial threat. This project aims to identify fraudulent transactions using machine learning techniques. The dataset used is highly imbalanced, with frauds being rare compared to normal transactions — making this both a challenging and important task.

---

## 📂 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:**
  - `Time`, `Amount`
  - `V1` to `V28` (PCA-anonymized features)
  - `Class`: `0` = Legitimate, `1` = Fraud

---

## 🧠 Technologies & Tools

- Python 🐍
- NumPy, Pandas
- Scikit-learn
- XGBoost / RandomForest / Logistic Regression
- Matplotlib, Seaborn
- SHAP / LIME (for model explainability)
- (Optional for advanced users) Streamlit / Flask for deployment

---

## 🛠️ Project Workflow

1. **Data Preprocessing**
   - Handling missing/nulls (if any)
   - Scaling `Amount` and `Time`
   - Balancing the dataset (SMOTE, under/oversampling)

2. **Exploratory Data Analysis**
   - Class distribution
   - Correlation heatmap
   - Fraud patterns

3. **Model Building**
   - Logistic Regression
   - Random Forest
   - XGBoost (Best performer)
   - Evaluation using Precision, Recall, F1 Score, ROC AUC

4. **Explainability (Optional)**
   - SHAP values to interpret why a transaction is fraud

5. **Deployment (Optional)**
   - Model saved using `joblib`
   - Simple web app using Streamlit or Flask (optional demo)

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/creditcard-fraud-detection.git
   cd creditcard-fraud-detection
