"""Transform module for the Telco Customer Churn TFX pipeline.

Defines the preprocessing_fn used by the TFX Transform component to encode
categorical features (vocabulary + one-hot) and scale numerical features to [0, 1].
"""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "InternetService": 3,
    "SeniorCitizen": 2,
    "PaperlessBilling": 2,
    "Partner": 2,
    "PhoneService": 2,
    "StreamingTV": 3,
    "gender": 2
}

NUMERICAL_FEATURES = [
    "MonthlyCharges",
    "TotalCharges",
    "tenure"
]

LABEL_KEY = "Churn"


def transformed_name(key):
    """Return the transformed feature name by appending '_xf' suffix.

    Args:
        key: Original feature name string.

    Returns:
        Feature name with '_xf' appended.
    """
    return key + "_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert an integer label tensor into a one-hot encoded vector.

    Args:
        label_tensor: Integer tensor of vocabulary indices (shape: [batch]).
        num_labels: Number of output classes (one-hot vector width).

    Returns:
        2-D float tensor of shape [batch, num_labels].
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """Preprocess raw input features into transformed features for training.

    Categorical features are mapped to vocabulary indices and then one-hot
    encoded to vectors of width num_labels+1. Numerical features are scaled
    to [0, 1]. The label is cast to int64.

    Args:
        inputs: Dict mapping feature key strings to raw feature tensors.

    Returns:
        Dict mapping transformed feature key strings to output tensors.
    """
    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
