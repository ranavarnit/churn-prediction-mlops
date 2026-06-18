# 🏦 Customer Churn Prediction with MLOps

## Overview
End-to-end machine learning pipeline for predicting customer churn in telecom industry.
Achieves **92% AUC** using XGBoost with automated MLflow tracking and FastAPI deployment.

## 🚀 Features
- **Data Processing**: Automated feature engineering with 17 engineered features
- **Model Training**: 4 algorithms compared (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Production API**: FastAPI deployment with &lt;100ms latency
- **Containerization**: Docker support for easy deployment

## 🛠 Tech Stack
- Python 3.9+
- scikit-learn, XGBoost
- MLflow
- FastAPI, Uvicorn
- Docker

## 📊 Results
| Model | AUC | F1-Score | CV AUC |
|-------|-----|----------|--------|
| XGBoost | 0.92 | 0.85 | 0.91 |
| Random Forest | 0.89 | 0.82 | 0.88 |
| Gradient Boosting | 0.88 | 0.81 | 0.87 |
| Logistic Regression | 0.82 | 0.75 | 0.80 |

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/varnit-rana/churn-prediction-mlops.git

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python churn_prediction_pipeline.py

# Start API server
uvicorn churn_prediction_pipeline:app --reload
