import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tfx.v1 as tfx

from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Trainer, Evaluator, Pusher
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel 
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration import metadata

# Konfigurasi
USERNAME = 'andyid23'
PIPELINE_NAME = f"{USERNAME}-pipeline"
PIPELINE_ROOT = os.path.join('.', PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, 'metadata', 'metadata.db')
DATA_ROOT = r'C:\Users\Dragon\Documents\github\andyid23\mlops\deploy\data'
SERVING_MODEL_DIR = 'serving_model'

print(f"Starting pipeline: {PIPELINE_NAME}")

# Definisikan komponen
example_gen = CsvExampleGen(input_base=DATA_ROOT)
statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema'])
transform = Transform(examples=example_gen.outputs['examples'], schema=schema_gen.outputs['schema'], module_file='modules/transform.py')
trainer = Trainer(module_file='modules/trainer.py', examples=transform.outputs['transformed_examples'], transform_graph=transform.outputs['transform_graph'], schema=schema_gen.outputs['schema'], train_args=trainer_pb2.TrainArgs(num_steps=2000), eval_args=trainer_pb2.EvalArgs(num_steps=500))

model_resolver = Resolver(
    strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('latest_blessed_model_resolver')

# ... (kode komponen lainnya sama) ...

evaluator = Evaluator(
    examples=example_gen.outputs["examples"],
    model=trainer.outputs["model"],
    baseline_model=model_resolver.outputs["model"],
    eval_config=tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Churn")], # Nama kolom asli di CSV
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name="BinaryAccuracy"),
                tfma.MetricConfig(class_name="AUC"),
            ])
        ],
        slicing_specs=[tfma.SlicingSpec()],
    ),
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=SERVING_MODEL_DIR))
)

components = [example_gen, statistics_gen, schema_gen, example_validator, transform, trainer, model_resolver, evaluator, pusher]

pipeline = tfx.dsl.Pipeline(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    components=components,
    metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH),
    enable_cache=True,
)

print("Running pipeline...")
BeamDagRunner().run(pipeline)
print("✅ Pipeline completed successfully!")