# Telco Customer Churn Prediction — MLOps Pipeline

**Dicoding Username:** andyid

---

## Problem Description

Customer churn is a critical business problem for telecom companies. This project builds
an end-to-end MLOps pipeline that trains a binary classification model to predict whether
a customer will churn (leave the service), enabling proactive retention strategies.

---

## Dataset

| Property | Value |
|---|---|
| Name | Telco Customer Churn |
| Source | IBM Sample Dataset |
| Rows | 7,043 customers |
| Features | 21 (demographic, service, billing) |
| Target | `Churn` (binary: Yes / No) |

Key features used by the model:

- **Categorical:** `InternetService`, `SeniorCitizen`, `PaperlessBilling`, `Partner`, `PhoneService`, `StreamingTV`, `gender`
- **Numerical:** `MonthlyCharges`, `TotalCharges`, `tenure`

---

## Solution Approach

The solution uses a **TFX (TensorFlow Extended)** pipeline for reproducible, production-ready ML:

```
CsvExampleGen → StatisticsGen → SchemaGen → ExampleValidator
     → Transform → Trainer → Resolver → Evaluator → Pusher
```

### Pipeline Steps

1. **CsvExampleGen** — Ingests raw CSV data and splits into train (80%) / eval (20%)
2. **StatisticsGen** — Computes descriptive statistics for data validation
3. **SchemaGen** — Infers the feature schema automatically
4. **ExampleValidator** — Detects anomalies and missing values
5. **Transform** — One-hot encodes categorical features; scales numericals to [0, 1]
6. **Trainer** — Trains a 3-layer DNN (128 → 64 → 32 units) with Dropout
7. **Resolver** — Retrieves the latest blessed model for comparison
8. **Evaluator** — Evaluates with BinaryAccuracy, AUC, Precision, Recall, FP, FN
9. **Pusher** — Pushes the blessed model to `serving_model/` for TF Serving

### Model Architecture

```
Input (concatenated features)
  → Dense(128, relu) → Dropout(0.3)
  → Dense(64, relu)  → Dropout(0.3)
  → Dense(32, relu)  → Dropout(0.3)
  → Dense(1, sigmoid)   # churn probability
```

- **Loss:** binary_crossentropy
- **Optimizer:** Adam (lr=0.001)
- **Metrics:** BinaryAccuracy, AUC

---

## Deployment

| Component | Technology |
|---|---|
| Model serving | TensorFlow Serving |
| Containerisation | Docker |
| Hosting platform | Railway |
| Monitoring | Prometheus |

**Serving URL:** `http://localhost:8501` *(update after Railway deployment)*

---

## Monitoring

Prometheus scrapes TF Serving metrics from the `/monitoring/prometheus/metrics` endpoint
every 5 seconds (global interval: 15 s).

Monitoring configuration files:
- `config/prometheus.config` — TF Serving Prometheus config
- `monitoring/prometheus.yml` — Prometheus scrape config
- `monitoring/Dockerfile` — Prometheus container

Screenshot reference: `andyid-monitoring.png`

---

## Running the Pipeline Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the TFX pipeline
python andyid-pipeline/local_pipeline.py
```

The pipeline artefacts (metadata, transformed data, model) are stored under
`andyid_pipeline/`. The blessed model is pushed to `serving_model/andyid-model/`.

---

## Running Prediction Tests

Open `andyid-testing.ipynb` in Jupyter and run all cells to send a sample prediction
request to the TF Serving REST endpoint and inspect the churn probability output.

---

## Project Structure

```
deploy-ops/
├── andyid-pipeline/
│   └── local_pipeline.py       # TFX pipeline definition & runner
├── modules/
│   ├── components.py           # Shared feature constants
│   ├── transform.py            # TFX Transform preprocessing_fn
│   └── trainer.py              # TFX Trainer run_fn
├── data/
│   └── Telco-Customer-Churn.csv.csv
├── serving_model/
│   └── andyid-model/               # Populated by Pusher after pipeline run
├── config/
│   └── prometheus.config       # TF Serving monitoring config
├── monitoring/
│   ├── prometheus.yml          # Prometheus scrape config
│   └── Dockerfile              # Prometheus container
├── Dockerfile                  # TF Serving container
├── andyid-testing.ipynb        # Prediction request test notebook
└── requirements.txt
```
