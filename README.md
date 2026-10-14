# AI-Powered Predictive Maintenance & Failure Diagnosis System

A machine-learning project that predicts potential industrial machine failures from operating parameters and estimates the likely failure mode. It combines binary failure prediction, multi-class diagnosis, model evaluation, threshold tuning, SHAP explainability, and simulation examples.

> **Project type:** Engineering / Machine Learning mini project  
> **Dataset:** UCI AI4I 2020 Predictive Maintenance  
> **Notebook:** Google Colab / Python

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Machine Learning Models](#machine-learning-models)
- [Reported Results](#reported-results)
- [Failure Modes](#failure-modes)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Example Prediction](#example-prediction)
- [Limitations and Future Work](#limitations-and-future-work)
- [Disclaimer](#disclaimer)

## Overview

Unexpected machine failures can cause downtime, maintenance costs, and production disruption. This project explores how machine-learning models can use operating data to flag possible failures and classify their probable type.

The project has two prediction tasks:

1. **Binary prediction:** estimate whether a machine failure is likely (`Failure` / `No Failure`).
2. **Multi-class diagnosis:** predict a failure category such as HDF, OSF, PWF, TWF, or RNF.

It also includes exploratory data analysis (EDA), engineering-oriented feature creation, model comparison, decision-threshold tuning, SHAP-based explainability, and example predictions for simulated machine inputs.

## Objectives

- Prepare and explore the AI4I 2020 predictive-maintenance dataset.
- Engineer features from temperature, speed, torque, and tool-wear measurements.
- Compare Logistic Regression, Random Forest, and XGBoost for binary failure prediction.
- Evaluate models with precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Explore threshold tuning for a recall-oriented maintenance use case.
- Train a multi-class XGBoost model for failure diagnosis.
- Use SHAP to investigate model explanations.
- Save trained models and related artifacts for reuse.

## Features

- Data loading from the UCI repository, with a synthetic-data fallback in the notebook.
- Missing-value and dataset overview checks.
- Failure-distribution charts, sensor boxplots, and a correlation heatmap.
- Feature engineering for temperature difference, mechanical power, and wear–torque interaction.
- Class-imbalance handling with class weights / sample weights.
- Comparison of three binary classifiers.
- Decision-threshold tuning.
- Multi-class failure diagnosis and confusion-matrix visualization.
- SHAP explainability plots.
- A reusable `predict_machine(...)` function for example inputs.
- Model and preprocessing artifact serialization with `joblib`.

## Technology Stack

- **Python**
- **Pandas** and **NumPy** — data manipulation and numerical operations
- **Matplotlib** and **Seaborn** — visualization
- **scikit-learn** — preprocessing, model training, splitting, and evaluation
- **XGBoost** — gradient-boosted decision trees
- **SHAP** — model explainability
- **ucimlrepo** — dataset retrieval from UCI
- **joblib** — saving trained artifacts

## Dataset

The project uses the [UCI AI4I 2020 Predictive Maintenance dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance).

The supplied project report records the following dataset profile:

| Item | Reported value |
|---|---:|
| Rows | 10,000 |
| Original columns | 12 |
| Missing values | 0 |
| Overall machine failure rate | 3.39% |
| Training records | 8,000 |
| Test records | 2,000 |
| Engineered features | 9 |

The notebook attempts to download the dataset from UCI. If retrieval fails, it generates synthetic data for demonstration. **Synthetic fallback data is not real machine telemetry and should not be used to claim real-world performance.**

### Input parameters

The prediction function accepts:

- Machine type (`L`, `M`, or `H`)
- Air temperature (K)
- Process temperature (K)
- Rotational speed (RPM)
- Torque (Nm)
- Tool wear (minutes)

The notebook derives additional features from these inputs.

## Workflow

```text
UCI AI4I 2020 dataset
        |
        v
Data loading and quality checks
        |
        v
Exploratory data analysis
        |
        v
Feature engineering
        |
        v
Stratified train/test split
        |
        v
Binary model comparison
        |
        v
Decision-threshold tuning
        |
        v
Multi-class failure diagnosis
        |
        v
SHAP explainability
        |
        v
Simulation and saved model artifacts
```

## Machine Learning Models

### Binary failure prediction

The notebook compares:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Class imbalance is addressed using class weighting and XGBoost's `scale_pos_weight`. The selected operating threshold is tuned to emphasize recall, since missing a failure can be costly in a maintenance setting.

### Multi-class failure diagnosis

A multi-class XGBoost classifier predicts a failure category. Sample weights are used to reduce the impact of class imbalance.

### Explainability

SHAP is included to help inspect how input features influence model outputs. Feature-level rankings should only be reported after reviewing the actual SHAP plots produced by the notebook.

## Reported Results

The following values are taken from the supplied project report and represent the notebook's reported evaluation results. They are not a guarantee of performance on new machines or live industrial systems.

### Binary model comparison

| Model | CV F1 | Test F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.2728 | 0.2963 | 0.1780 | 0.8824 | 0.9393 | 0.4276 |
| Random Forest | 0.8644 | 0.8235 | 0.9608 | 0.7206 | 0.9610 | 0.8749 |
| XGBoost | 0.8279 | **0.8485** | 0.8750 | 0.8235 | **0.9817** | **0.8808** |

XGBoost had the highest reported base test F1-score, ROC-AUC, and PR-AUC among the three compared models.

### Recall-oriented threshold

The report records a tuned threshold of `0.0835`:

| Metric | Reported value |
|---|---:|
| F1-score | 0.7484 |
| Recall | 0.8529 |
| Precision | 0.6667 |

Reported confusion-matrix counts:

| | Predicted failure | Predicted no failure |
|---|---:|---:|
| Actual failure | 58 | 10 |
| Actual no failure | 29 | 1,903 |

This operating point captures more failures while accepting more false alarms. The appropriate threshold depends on validated operational requirements and the relative costs of missed failures and unnecessary inspections.

### Multi-class diagnosis

The report records **98% overall test accuracy**, but aggregate accuracy does not tell the full story for rare failure categories.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| HDF | 0.93 | 0.97 | 0.95 | 29 |
| No Failure | 0.99 | 0.99 | 0.99 | 1,930 |
| OSF | 0.93 | 0.93 | 0.93 | 15 |
| PWF | 0.92 | 1.00 | 0.96 | 12 |
| RNF | 0.00 | 0.00 | 0.00 | 4 |
| TWF | 0.00 | 0.00 | 0.00 | 10 |

RNF and TWF were not reliably identified in the reported multi-class test results. This is an important limitation despite the high overall accuracy.

## Failure Modes

The dataset uses the following labels:

| Label | Meaning |
|---|---|
| **TWF** | Tool Wear Failure |
| **HDF** | Heat Dissipation Failure |
| **PWF** | Power Failure |
| **OSF** | Overstrain Failure |
| **RNF** | Random Failure |

The dataset's failure-mode labels are used as supplied by the AI4I dataset. Refer to the dataset documentation for its detailed definitions.

## Project Structure

A suggested GitHub repository layout is:

```text
predictive-maintenance-failure-diagnosis/
├── AI_Predictive_Maintenance_Failure_Diagnosis.ipynb
├── ai_predictive_maintenance_failure_diagnosis.py
├── README.md
├── requirements.txt
├── data/
│   └── (optional local dataset; do not commit large or restricted files)
└── outputs/
    └── (generated model artifacts; optional)
```

Your repository may use different filenames. Keep the README paths consistent with the files you actually upload. The notebook currently saves generated data and model artifacts under `/content`, which is typical for Google Colab.

## Getting Started

### Option 1: Run in Google Colab

1. Upload the notebook to Google Colab.
2. Run the cells in order.
3. Allow the notebook to install or import required packages.
4. Check the printed evaluation metrics and generated plots.
5. Download any generated artifacts you want to keep.

### Option 2: Run locally

Use Python 3.10 or another version compatible with your installed ML libraries.

1. Clone your repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repository>.git
   cd <your-repository>
   ```

2. Create and activate a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **Linux / macOS**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Python script if it is compatible with your local environment:

   ```bash
   python ai_predictive_maintenance_failure_diagnosis.py
   ```

   The supplied script contains Google Colab/Jupyter-specific commands such as `!pip install` and `display(...)`. If running it as a plain `.py` file, convert those notebook-specific commands to standard Python first, or run the notebook in Colab/Jupyter.

### Suggested `requirements.txt`

Create a `requirements.txt` file with:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
ucimlrepo
joblib
```

For reproducible results, pin versions after confirming which versions work in your environment.

## Example Prediction

The notebook defines a function with this interface:

```python
predict_machine(
    machine_type,
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear
)
```

Example inputs used in the notebook:

```python
# Nominal operating example
predict_machine('L', 300, 311, 1400, 42, 50)

# HDF-labelled simulation scenario
predict_machine('L', 302, 311, 1350, 25, 120)

# Extreme-wear simulation scenario
predict_machine('H', 298, 308, 1450, 75, 230)
```

The supplied report records these simulation outputs:

| Scenario | Failure probability | Risk label | Diagnosis |
|---|---:|---|---|
| Nominal machine | 0.0000 | Low | No Failure |
| HDF-run test | 0.0012 | Low | No Failure |
| Extreme-wear machine | 0.9993 | High | TWF |

These are predictions for specific example inputs, not observed future outcomes. In particular, the HDF-labelled simulation input was reported as **No Failure** by the model.

## Limitations and Future Work

- **Rare failure classes:** RNF and TWF had weak results in the reported multi-class evaluation.
- **Risk-band logic:** The current risk-label logic makes the `Medium` category unreachable. Review and correct the risk thresholds before using this function operationally.
- **Synthetic fallback:** Synthetic data is only a demonstration fallback and does not validate real-world performance.
- **Evaluation:** Validate the model on new, representative machine data before relying on its predictions.
- **Threshold selection:** Choose thresholds based on measured maintenance costs and operational requirements.
- **Explainability:** Add and document feature-level SHAP results from actual notebook outputs.
- **Production readiness:** Add input validation, model/version tracking, monitoring, drift detection, logging, and deployment tests before production use.

## Disclaimer

This repository is an educational engineering and machine-learning project. It is **not a certified industrial safety system** and should not be used as the sole basis for maintenance, safety, or shutdown decisions. Model outputs are estimates and require validation against representative operational data.

## Acknowledgements

- [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance)
- The open-source Python data science and machine-learning ecosystem, including scikit-learn, XGBoost, and SHAP.
