"""Trainer module for Telco Customer Churn prediction model.

Defines the TFX run_fn entry point used by the Trainer component, along with
helper functions for building the Keras model, creating input datasets, and
exporting a TensorFlow Serving-compatible signature.
"""

import tensorflow as tf
import tensorflow_transform as tft
from keras.layers import Dense, Dropout, Input

# ── Feature constants (duplicated from transform.py because TFX packages
#    trainer.py into a standalone wheel that cannot import modules.transform) ──

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
    """Return the transformed feature name by appending '_xf' suffix."""
    return key + "_xf"

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DENSE_DROPOUT = 0.3


def _input_fn(file_pattern, tf_transform_output, batch_size):
    """Membuat dataset untuk training/evaluasi dari TFRecord (GZIP-compressed)."""
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=lambda path: tf.data.TFRecordDataset(path, compression_type='GZIP'),
        label_key=transformed_name(LABEL_KEY)
    )
    return dataset

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Return a serving function compatible with TensorFlow Serving REST/gRPC.

    The returned function accepts a batch of serialised tf.train.Example bytes,
    parses them using the raw (pre-transform) feature spec, applies the
    transform layer, runs the model, and returns the prediction output.

    Args:
        model: Trained Keras model with an attached tft_layer attribute.
        tf_transform_output: TFTransformOutput object for the transform graph.

    Returns:
        A @tf.function suitable for use as a SavedModel serving signature.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove the label key from the serving spec — clients never send labels.
    raw_feature_spec.pop(LABEL_KEY, None)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Parse and serve a batch of serialised tf.train.Example protos."""
        raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = model.tft_layer(raw_features)
        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _build_model_inputs(tf_transform_output):  # pylint: disable=unused-argument
    """Create Keras Input layers matching the shapes produced by preprocessing_fn.

    Numerical features are scaled scalars (shape=(1,), float32).
    Categorical features are one-hot vectors of width num_labels+1 (float32).

    Args:
        tf_transform_output: TFTransformOutput (kept for API consistency).

    Returns:
        Dict mapping transformed feature name → tf.keras.Input layer.
    """
    inputs = {}
    for feature in NUMERICAL_FEATURES:
        inputs[transformed_name(feature)] = Input(
            shape=(1,),
            name=transformed_name(feature),
            dtype=tf.float32
        )
    for feature, num_labels in CATEGORICAL_FEATURES.items():
        inputs[transformed_name(feature)] = Input(
            shape=(num_labels + 1,),
            name=transformed_name(feature),
            dtype=tf.float32
        )
    return inputs


def _build_model(inputs, tf_transform_output):  # pylint: disable=unused-argument
    """Build and compile the Keras DNN classification model.

    Args:
        inputs: Dict of Keras Input layers from _build_model_inputs.
        tf_transform_output: TFTransformOutput (kept for API consistency).

    Returns:
        Compiled tf.keras.Model.
    """
    denses = []
    for feature in NUMERICAL_FEATURES:
        denses.append(inputs[transformed_name(feature)])
    for feature in CATEGORICAL_FEATURES:
        denses.append(inputs[transformed_name(feature)])

    concatenated = tf.concat(denses, axis=-1)

    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(DENSE_DROPOUT)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(DENSE_DROPOUT)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(DENSE_DROPOUT)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    model.summary()
    return model


def run_fn(fn_args):
    """Train the model and save it."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Load dataset
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, BATCH_SIZE)

    # Build model
    inputs = _build_model_inputs(tf_transform_output)
    model = _build_model(inputs, tf_transform_output)

    # Hitung steps_per_epoch
    steps_per_epoch = fn_args.train_steps if fn_args.train_steps else 100
    validation_steps = fn_args.eval_steps if fn_args.eval_steps else 50

    print(f"🚀 Training model dengan steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

    # Train model
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=eval_dataset,
        validation_steps=validation_steps,
        verbose=1
    )

    # Save model
    serving_fn = _get_serve_tf_examples_fn(model, tf_transform_output)
    model.save(fn_args.serving_model_dir, save_format='tf', signatures={
        'serving_default': serving_fn
    })
    print(f"✅ Model berhasil disimpan di: {fn_args.serving_model_dir}")