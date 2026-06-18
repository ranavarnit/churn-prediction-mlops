# churn_prediction_pipeline.py
# End-to-end MLOps pipeline for customer churn prediction
# Includes: data preprocessing, model training, experiment tracking, 
# model registry, and FastAPI deployment

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    confusion_matrix, f1_score, roc_curve
)
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# For API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class ChurnPredictor:
    def __init__(self, experiment_name="churn_prediction_v1"):
        self.experiment_name = experiment_name
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        self.metrics = {}
        mlflow.set_experiment(experiment_name)

    def generate_synthetic_data(self, n_samples=10000):
        np.random.seed(42)
        data = {
            'customer_id': range(10000, 10000 + n_samples),
            'tenure': np.random.randint(0, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'contract_length': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                               n_samples, p=[0.55, 0.25, 0.20]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                               'Bank transfer', 'Credit card'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'num_support_tickets': np.random.poisson(2, n_samples),
            'avg_call_duration': np.random.uniform(5, 30, n_samples),
            'num_products': np.random.randint(1, 5, n_samples),
            'satisfaction_score': np.random.randint(1, 6, n_samples),
        }
        df = pd.DataFrame(data)
        churn_prob = (
            0.3 * (df['tenure'] < 12) +
            0.2 * (df['contract_length'] == 'Month-to-month') +
            0.15 * (df['payment_method'] == 'Electronic check') +
            0.1 * (df['tech_support'] == 'No') +
            0.1 * (df['num_support_tickets'] > 3) +
            0.1 * (df['satisfaction_score'] <= 2) +
            0.05 * (df['monthly_charges'] > 80)
        )
        df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
        return df

    def feature_engineering(self, df, is_training=True):
        df = df.copy()
        df['contract_length_encoded'] = df['contract_length'].map({
            'Month-to-month': 0, 'One year': 1, 'Two year': 2
        })
        df['payment_risk'] = df['payment_method'].apply(lambda x: 1 if x == 'Electronic check' else 0)
        df['has_internet'] = (df['internet_service'] != 'No').astype(int)
        df['has_fiber'] = (df['internet_service'] == 'Fiber optic').astype(int)
        df['charges_per_tenure'] = df['total_charges'] / (df['tenure'] + 1)
        df['monthly_to_total_ratio'] = df['monthly_charges'] / (df['total_charges'] + 1)
        df['risk_score'] = (
            df['payment_risk'] * 0.3 +
            (df['tech_support'] == 'No').astype(int) * 0.2 +
            (df['online_security'] == 'No').astype(int) * 0.2 +
            (df['num_support_tickets'] > 3).astype(int) * 0.15 +
            (df['satisfaction_score'] <= 2).astype(int) * 0.15
        )
        feature_cols = [
            'tenure', 'monthly_charges', 'total_charges', 'contract_length_encoded',
            'payment_risk', 'has_internet', 'has_fiber', 'tech_support',
            'online_security', 'paperless_billing', 'num_support_tickets',
            'avg_call_duration', 'num_products', 'satisfaction_score',
            'charges_per_tenure', 'monthly_to_total_ratio', 'risk_score'
        ]
        df['tech_support'] = (df['tech_support'] == 'Yes').astype(int)
        df['online_security'] = (df['online_security'] == 'Yes').astype(int)
        df['paperless_billing'] = (df['paperless_billing'] == 'Yes').astype(int)
        return df[feature_cols], df['churn'] if 'churn' in df.columns else None

    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models_config = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.01]}
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {'n_estimators': [100, 200], 'max_depth': [3, 5]}
            }
        }

        best_auc = 0
        for name, config in models_config.items():
            with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                y_pred = grid_search.predict(X_test_scaled)
                y_pred_proba = grid_search.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                cv_scores = cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric('test_auc', auc)
                mlflow.log_metric('test_f1', f1)
                mlflow.log_metric('cv_auc_mean', cv_scores.mean())
                mlflow.log_metric('cv_auc_std', cv_scores.std())
                mlflow.sklearn.log_model(grid_search.best_estimator_, name)
                self.models[name] = {'model': grid_search.best_estimator_, 'auc': auc, 'f1': f1, 'cv_mean': cv_scores.mean(), 'params': grid_search.best_params_}
                if auc > best_auc:
                    best_auc = auc
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = name
                print(f"{name}: AUC={auc:.4f}, F1={f1:.4f}, CV_AUC={cv_scores.mean():.4f}")

        joblib.dump(self.best_model, 'best_churn_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = dict(zip(X.columns, np.abs(self.best_model.coef_[0])))
        self.metrics = {'best_model': self.best_model_name, 'test_auc': best_auc, 'all_models': {k: {'auc': v['auc'], 'f1': v['f1']} for k, v in self.models.items()}}
        return self.metrics

    def get_feature_importance(self):
        if self.feature_importance:
            return sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return None

# FastAPI Application
app = FastAPI(title="Churn Prediction API", version="1.0")

class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_length: str
    payment_method: str
    internet_service: str
    tech_support: str
    online_security: str
    paperless_billing: str
    num_support_tickets: int
    avg_call_duration: float
    num_products: int
    satisfaction_score: int

predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    predictor = ChurnPredictor()
    predictor.best_model = joblib.load('best_churn_model.pkl')
    predictor.scaler = joblib.load('scaler.pkl')

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    try:
        data = pd.DataFrame([customer.dict()])
        X, _ = predictor.feature_engineering(data, is_training=False)
        X_scaled = predictor.scaler.transform(X)
        churn_prob = predictor.best_model.predict_proba(X_scaled)[0, 1]
        prediction = int(churn_prob > 0.5)
        risk_level = 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
        return {'churn_probability': float(churn_prob), 'churn_prediction': prediction, 'risk_level': risk_level, 'confidence': float(abs(churn_prob - 0.5) * 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

if __name__ == "__main__":
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - MLOps PIPELINE")
    print("=" * 60)
    cp = ChurnPredictor()
    print("
[1] Generating synthetic customer data...")
    df = cp.generate_synthetic_data(n_samples=10000)
    print(f"     Dataset shape: {df.shape}")
    print(f"     Churn rate: {df['churn'].mean():.2%}")
    print("
[2] Feature engineering...")
    X, y = cp.feature_engineering(df)
    print(f"     Features: {list(X.columns)}")
    print("
[3] Training models with MLflow tracking...")
    metrics = cp.train_and_evaluate(X, y)
    print(f"
[4] Best Model: {metrics['best_model']}")
    print(f"     Test AUC: {metrics['test_auc']:.4f}")
    print("
[5] Top Features:")
    for feat, imp in cp.get_feature_importance()[:5]:
        print(f"     {feat}: {imp:.4f}")
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("
[6] Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
