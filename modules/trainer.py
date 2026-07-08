"""Modul training khusus untuk dataset Telco-Customer-Churn."""
import tensorflow as tf
import tensorflow_transform as tft
from modules.transform import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    _transformed_name,
)

def _get_keras_model():
    """Membangun arsitektur model klasifikasi biner."""
    inputs = []
    
    # Input untuk fitur numerik (shape=1)
    for feature in NUMERICAL_FEATURES:
        inputs.append(tf.keras.layers.Input(
            name=_transformed_name(feature), shape=(1,), dtype=tf.float32
        ))
        
    # Input untuk fitur kategorikal (shape=1, karena sudah di-encode jadi integer)
    for feature in CATEGORICAL_FEATURES:
        inputs.append(tf.keras.layers.Input(
            name=_transformed_name(feature), shape=(1,), dtype=tf.int64
        ))

    # Gabungkan semua input
    concat = tf.keras.layers.concatenate(inputs)
    
    # Lapisan Dense (Deep Learning)
    x = tf.keras.layers.Dense(128, activation="relu")(concat)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer (1 neuron dengan sigmoid untuk klasifikasi biner)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")]
    )
    return model

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Membuat input function untuk training dan evaluation."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key=_transformed_name(LABEL_KEY),
    )
    return dataset

def run_fn(fn_args):
    """Fungsi utama yang dipanggil oleh TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 32)

    model = _get_keras_model()
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10, # Bisa ditambah jika akurasi masih rendah
    )

    model.save(fn_args.serving_model_dir, save_format="tf")