"""TFX local pipeline for Telco Customer Churn prediction.

This module defines the TFX pipeline components and orchestrates them using
BeamDagRunner for local execution.
"""

import os
import logging

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.orchestration import metadata
from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import standard_artifacts

# Setup Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# --- KONFIGURASI PATH ---
DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/transform.py'
TRAINER_MODULE_FILE = 'modules/trainer.py'
SERVING_MODEL_DIR = os.path.join(os.getcwd(), 'serving_model', 'andyid-model')

# Nama pipeline harus sesuai username Dicoding Anda
PIPELINE_NAME = 'andyid_pipeline'
PIPELINE_ROOT = os.path.join(os.getcwd(), PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata', 'metadata.db')


def init_components(data_root, transform_module, trainer_module):
    """Inisialisasi semua komponen TFX.

    Args:
        data_root: Path ke direktori dataset CSV.
        transform_module: Path ke modul transform.py.
        trainer_module: Path ke modul trainer.py.

    Returns:
        List dari 9 komponen TFX yang telah dikonfigurasi.
    """
    # 1. ExampleGen
    example_gen = tfx.components.CsvExampleGen(
        input_base=data_root,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
            ])
        )
    )

    # 2. StatisticsGen
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # 3. SchemaGen
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )

    # 4. ExampleValidator
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # 5. Transform
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )

    # 6. Trainer
    trainer = tfx.components.Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=None,
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    # 7. Resolver (untuk mengambil model terbaik)
    resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=standard_artifacts.ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # 8. Evaluator
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=resolver.outputs['model'],
        eval_config=tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(label_key='Churn', prediction_key='outputs')
            ],
            metrics_specs=[
                tfma.MetricsSpec(metrics=[
                    tfma.MetricConfig(
                        class_name='BinaryAccuracy',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.5}
                            )
                        )
                    ),
                    tfma.MetricConfig(
                        class_name='AUC',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.5}
                            )
                        )
                    ),
                    tfma.MetricConfig(class_name='Precision'),
                    tfma.MetricConfig(class_name='Recall'),
                    tfma.MetricConfig(class_name='FalsePositives'),
                    tfma.MetricConfig(class_name='FalseNegatives')
                ])
            ],
            slicing_specs=[
                tfma.SlicingSpec()
            ]
        )
    )

    # 9. Pusher
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        )
    )

    return [
        example_gen, statistics_gen, schema_gen, example_validator,
        transform, trainer, resolver, evaluator, pusher
    ]


def init_pipeline(pipeline_name, pipeline_root, metadata_path):
    """Membuat dan menjalankan pipeline TFX dengan BeamDagRunner.

    Args:
        pipeline_name: Nama pipeline (sesuai username Dicoding).
        pipeline_root: Direktori root tempat artefak pipeline disimpan.
        metadata_path: Path ke file SQLite metadata.
    """
    components = init_components(DATA_ROOT, TRANSFORM_MODULE_FILE, TRAINER_MODULE_FILE)

    metadata_connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path
    )

    pipeline_config = Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata_connection_config,
        components=components,
        enable_cache=True
    )

    print(f"Menjalankan pipeline: {pipeline_name}")
    BeamDagRunner().run(pipeline_config)


# Jalankan pipeline jika file ini dieksekusi langsung
if __name__ == '__main__':
    init_pipeline(PIPELINE_NAME, PIPELINE_ROOT, METADATA_PATH)
