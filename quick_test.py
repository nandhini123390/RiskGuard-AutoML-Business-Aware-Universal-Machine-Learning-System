#!/usr/bin/env python3
"""
Quick test of the system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
import json
import os

def test_telco():
    """Test with telco churn data"""
    print("🧪 Testing with Telco Churn dataset...")
    
    # Load data
    df = pd.read_csv('demos/telco_churn/data.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Check target
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column not found")
        return
    
    # Simple preprocessing
    X = df.drop(columns=['Churn', 'customerID'])
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Convert categorical columns
    for col in X.select_dtypes(['object']).columns:
        X[col] = pd.factorize(X[col])[0]
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train simple model
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ Test complete!")
    print(f"   AUC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    
    # Save results
    os.makedirs('outputs/quick_test', exist_ok=True)
    results = {
        'dataset': 'telco_churn',
        'model': 'XGBoost',
        'auc': float(auc),
        'accuracy': float(acc),
        'test_size': len(X_test)
    }
    
    with open('outputs/quick_test/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    test_telco()
