"""
Pre-Insolvency Distress Score (PIDS) - Rule‑Based Only
No ML, no scikit-learn, no model.pkl needed.
"""

# Rule weights (based on backtesting)
WEIGHTS = {
    'gst_delayed_months': 10,
    'legal_cases_count': 5,
    'auditor_resigned': 15,
    'director_change_frequency': 8,
    'negative_net_worth': 12,
    'payment_days_outstanding': 7,
    'nclt_mention': 20
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
    raw_score = 0
    for key, value in features.items():
        if key in WEIGHTS:
            normalized = min(value / MAX_VALUES.get(key, 1), 1.0)
            raw_score += normalized * WEIGHTS[key]
    max_possible = sum(WEIGHTS.values())
    return min(100, (raw_score / max_possible) * 100)

def compute_dps(features_dict):
    """Public function: returns distress score 0-100"""
    return round(compute_rule_based_score(features_dict), 1)

def classify_risk(score):
    if score >= 70:
        return "CRITICAL - Immediate default likely"
    elif score >= 50:
        return "HIGH RISK - Distress visible, act within 3 months"
    elif score >= 30:
        return "MODERATE RISK - Monitor monthly"
    else:
        return "LOW RISK - Normal operations"
