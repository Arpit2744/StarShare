# Student Performance Indicator — ML System
## Problem Statement

This system trains and evaluates regression models to predict student math performance based on demographic and academic attributes.

The goal is not to expose an API or UI, but to produce a reproducible, testable, and auditable training pipeline that enforces strict data contracts and model validation before a model is accepted.

## System Scope

This repository contains only the training system.

It does not include:

Web applications
APIs
Docker images
Visualization dashboards
Experiment notebooks
Training and serving are separate concerns by design.

## Data Contract

All input data must conform to a strict schema defined in:
```
data/schema.yaml
```
The schema specifies:

Numerical features
Categorical features
Identifier columns (non-features)
Target column

## Enforcement Rules

Missing required columns → hard failure
Unexpected extra columns → logged warning
Target leakage → blocked
Schema mismatch → training aborts
No model trains on unvalidated data.

## Directory Structure
```
ml-system/
├── data/
│   ├── raw/                # Immutable input data
│   ├── processed/          # Derived data (reproducible)
│   └── schema.yaml         # Data contract
│
├── src/
│   ├── config.py           # Global configuration + schema loader
│   ├── features/
│   │   └── build.py        # Deterministic feature engineering
│   ├── train/
│   │   └── run.py          # Training entrypoint
│   ├── model/
│   │   └── registry.py     # Model persistence & versioning
│   └── validation/
│       └── checks.py       # Schema + invariants
│
├── tests/
│   ├── test_schema.py      # Data contract tests
│   ├── test_features.py    # Feature pipeline tests
│   └── test_training.py    # Training smoke test
│
├── pipelines/
│   └── train.sh            # Single execution path
│
├── requirements.txt
└── README.md
```

## Execution Path

Training is executed via one command only:
```
./pipelines/train.sh
```
This script:

Loads schema
Validates raw data
Builds deterministic features
Trains multiple candidate models
Selects the best model by R² score
Persists the model and preprocessor

If this script fails, the system is considered broken.

## Reproducibility Guarantees

Given:

The same raw dataset
The same schema
The same dependency versions

Running train.sh will:

Produce the same feature matrix
Evaluate the same model candidates
Select the same best model

## Model Selection

Multiple regressors are evaluated using cross-validated grid search.

Selection criteria:

Primary metric: R²
Models below threshold are rejected
Best model is persisted via the registry

No manual model picking.
No notebook-based decisions.

## Validation & Testing

Tests are mandatory.

#### What is tested

Schema enforcement
Feature pipeline consistency
Training pipeline executability

Run all test with:
```
pytest -q
```

if tests fail:
the system is not trusted
training result are invalid

## Failure Modes

#### Training will fail loudly if:

Schema does not match input data
Required columns are missing
Feature pipeline produces inconsistent shapes
Model training errors occur
Evaluation metrics fall below acceptable thresholds

Silent failure is treated as a bug.

## What This Repo Is Not

A demo
A tutorial
A notebook collection
A deployment artifact

this is a training system, not a product

## Final Note

If deleting:

notebooks
cached artifacts
logs

breaks this system, that is a bug.

If the system cannot explain how a model was trained without referencing a notebook, that is a design failure.