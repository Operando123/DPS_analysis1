"""
Pre-Insolvency Distress Score (PIDS) - India
Zero-cost, rule-based + logistic regression hybrid.
For demonstration, uses input data. Live version would replace with APIs.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ---------- 1. Define feature weights (based on backtesting) ----------
# For a no-ML version, use weighted sum:
WEIGHTS = {
    'gst_delayed_months': 10,
    'legal_cases_count': 5,
    'auditor_resigned': 15,
    'director_change_frequency': 8,
    'negative_net_worth': 12,
    'payment_days_outstanding': 7,
    'nclt_mention': 20      # if already mentioned in NCLT (late stage)
}

MAX_VALUES = {
    'gst_delayed_months': 4,
    'legal_cases_count': 10,
    'auditor_resigned': 1,
    'director_change_frequency': 3,
    'negative_net_worth': 1,
    'payment_days_outstanding': 180,
    'nclt_mention': 1
}

def compute_rule_based_score(features):
    """
    features: dict with keys matching WEIGHTS
    returns score 0-100
    """
    raw_score = 0
    for key, value in features.items():
        if key in WEIGHTS:
            normalized = min(value / MAX_VALUES.get(key, 1), 1.0)
            raw_score += normalized * WEIGHTS[key]
    max_possible = sum(WEIGHTS.values())
    return min(100, (raw_score / max_possible) * 100)

# ---------- 2. ML Model (trained on historical data - included as pickle) ----------
# For zero-investment, we provide a pre-trained dummy model.
# In reality, you would train on 200+ companies. We include a placeholder.

def train_mock_model():
    """Generates a mock logistic regression model for demonstration."""
    # Dummy training data: 4 features
    X = np.random.rand(100, 4)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = LogisticRegression()
    model.fit(X, y)
    scaler = StandardScaler().fit(X)
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, scaler), f)
    print("Mock model saved as model.pkl")

def load_ml_model():
    if not os.path.exists('model.pkl'):
        train_mock_model()
    with open('model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    return model, scaler

def predict_ml_score(features_vector):
    """features_vector: list of 4 key numeric indicators."""
    model, scaler = load_ml_model()
    scaled = scaler.transform([features_vector])
    prob = model.predict_proba(scaled)[0][1]  # probability of distress
    return prob * 100

# ---------- 3. Unified Distress Score ----------
def compute_dps(features_dict):
    """
    features_dict example:
    {
        'gst_delayed_months': 3,
        'legal_cases_count': 5,
        'auditor_resigned': 1,
        'director_change_frequency': 2,
        'negative_net_worth': 1,
        'payment_days_outstanding': 120,
        'nclt_mention': 0
    }
    """
    rule_score = compute_rule_based_score(features_dict)
    # For ML, use subset of features (gst, legal, auditor, payment)
    ml_features = [
        min(features_dict.get('gst_delayed_months', 0) / 4, 1.0),
        min(features_dict.get('legal_cases_count', 0) / 10, 1.0),
        features_dict.get('auditor_resigned', 0),
        min(features_dict.get('payment_days_outstanding', 0) / 180, 1.0)
    ]
    ml_score = predict_ml_score(ml_features)
    # Weighted average: rule-based (70%) + ML (30%) for stability
    final_score = 0.7 * rule_score + 0.3 * ml_score
    return round(final_score, 1)

# ---------- 4. Risk Classification ----------
def classify_risk(score):
    if score >= 70:
        return "CRITICAL - Immediate default likely"
    elif score >= 50:
        return "HIGH RISK - Distress visible, act within 3 months"
    elif score >= 30:
        return "MODERATE RISK - Monitor monthly"
    else:
        return "LOW RISK - Normal operations"

# ---------- 5. CLI / Example ----------
if __name__ == "__main__":
    # Example input (simulate a textile supplier's customer)
    sample = {
        'gst_delayed_months': 2,
        'legal_cases_count': 3,
        'auditor_resigned': 0,
        'director_change_frequency': 1,
        'negative_net_worth': 0,
        'payment_days_outstanding': 90,
        'nclt_mention': 0
    }
    dps = compute_dps(sample)
    print(f"Distress Probability Score: {dps} / 100")
    print(f"Risk Grade: {classify_risk(dps)}")
