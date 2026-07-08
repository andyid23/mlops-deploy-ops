"""Modul preprocessing khusus untuk dataset Telco-Customer-Churn."""
import tensorflow as tf
import tensorflow_transform as tft

# Kolom yang akan dibuang (tidak dipakai untuk training)
UNUSED_FEATURES = ["customerID"]

# Kolom Numerik (berupa angka)
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Kolom Kategorikal (berupa teks/kategori)
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# Kolom Target (Label)
LABEL_KEY = "Churn"

def _transformed_name(key):
    """Mengembalikan nama fitur yang sudah ditransformasi."""
    return f"{key}_xf"

def preprocessing_fn(inputs):
    """Fungsi preprocessing untuk TFX Transform."""
    outputs = inputs.copy()

    # 1. Normalisasi fitur numerik (Z-score)
    for feature in NUMERICAL_FEATURES:
        # TotalCharges seringkali punya tipe string/float yang campur aduk, kita cast ke float32
        outputs[_transformed_name(feature)] = tft.scale_to_z_score(
            tf.cast(inputs[feature], tf.float32)
        )

    # 2. Encode fitur kategorikal menjadi integer (Vocabulary)
    for feature in CATEGORICAL_FEATURES:
        outputs[_transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature], vocab_filename=feature
        )

    # 3. Encode Label (Churn: "Yes" -> 1, "No" -> 0)
    outputs[_transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], "Yes"), tf.float32
    )

    return outputs