"""Shared feature constants and helpers for the Telco Churn TFX pipeline.

This module is the single source of truth for feature definitions used by
transform.py, trainer.py, and any other pipeline modules.
"""

# Categorical features mapped to their vocabulary size (number of distinct labels,
# excluding the "unknown" bucket). The one-hot output width from preprocessing_fn
# is num_labels + 1 (to accommodate the unknown bucket added by
# tft.compute_and_apply_vocabulary).
CATEGORICAL_FEATURES = {
    "InternetService": 3,
    "SeniorCitizen": 2,
    "PaperlessBilling": 2,
    "Partner": 2,
    "PhoneService": 2,
    "StreamingTV": 3,
    "gender": 2
}

# Numerical features that are scaled to [0, 1] by preprocessing_fn.
NUMERICAL_FEATURES = [
    "MonthlyCharges",
    "TotalCharges",
    "tenure"
]

# Target column name (binary: 0 = no churn, 1 = churn).
LABEL_KEY = "Churn"


def transformed_name(key):
    """Return the transformed feature name by appending '_xf' suffix.

    Args:
        key: Original feature name string.

    Returns:
        Feature name with '_xf' appended.
    """
    return key + "_xf"
