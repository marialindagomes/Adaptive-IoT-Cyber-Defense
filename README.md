# AdaptiveExplain-IDS

## An Explainable and Adaptive Intrusion Detection Framework for IoT Networks


## Overview

AdaptiveExplain-IDS is an Explainable Artificial Intelligence (XAI)-based adaptive cyber threat detection framework designed for IoT network security.

The framework integrates machine learning-based intrusion detection, probability calibration, unknown attack detection, explainable AI, explanation reliability measurement, threat risk assessment, concept drift detection, adaptive retraining, and automated response generation.

The objective of this framework is to develop an intelligent intrusion detection system capable of:

- Detecting known cyber attacks
- Identifying unknown attack behaviour
- Explaining model decisions
- Measuring explanation reliability
- Adapting to changing traffic distributions
- Generating automated security responses


# Framework Workflow


```
Dataset Collection

        ↓

Dataset Inspection

        ↓

Data Preparation

        ↓

Feature Processing

        ↓

Train / Validation / Test Split

        ↓

Baseline Machine Learning Models

        ↓

XGBoost Threat Detector

        ↓

Probability Calibration

        ↓

Unknown Attack Detection

        ↓

Explainable AI (SHAP)

        ↓

Explanation Reliability Score (ERS)

        ↓

Threat Risk Assessment

        ↓

ERGAR Adaptive Response Engine

        ↓

Real-Time Streaming Simulation

        ↓

ADWIN Drift Detection

        ↓

Adaptive Retraining

        ↓

Cross Dataset Validation

        ↓

Final Evaluation
```


# Key Contributions


## 1. Machine Learning Based Threat Detection

The framework uses XGBoost as the primary threat detection model for IoT intrusion classification.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC


## 2. Probability Calibration

The framework improves prediction confidence using probability calibration techniques:

- Platt Scaling
- Isotonic Regression

Calibration performance is evaluated using:

- Brier Score
- Log Loss
- Accuracy


## 3. Unknown Attack Detection

The framework evaluates the capability of identifying previously unseen attack behaviour.

Controlled distribution changes are introduced to evaluate model robustness against unknown attack scenarios.


## 4. Real-Time Streaming Simulation

The framework supports continuous IoT traffic monitoring.

The streaming workflow is:

```
Network Traffic Stream

        ↓

Feature Processing

        ↓

Threat Prediction

        ↓

Drift Monitoring

        ↓

Adaptive Update
```


## 5. Concept Drift Detection

ADWIN (Adaptive Windowing) is used to detect changes in network traffic behaviour.

Workflow:

```
Traffic Distribution Change

        ↓

ADWIN Detection

        ↓

Drift Event

        ↓

Adaptive Retraining

        ↓

Updated Model
```


## 6. Adaptive Retraining

After drift detection, recent traffic samples are used to update the threat detection model.

This allows the framework to adapt to changing attack patterns.


## 7. Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) is integrated to explain XGBoost predictions.

The framework identifies the most influential features contributing to threat classification.

Example important features:

- rst_count
- Number
- Tot size
- flow_duration
- Header_Length


## 8. Explanation Reliability Score (ERS)

ERS measures the reliability of generated explanations.

The score considers:

- SHAP stability
- Feature contribution consistency
- Model confidence

ERS provides an additional reliability layer for AI-based cybersecurity decisions.


## 9. Threat Risk Assessment

The Risk Engine combines multiple security indicators:

```
Attack Probability

        +

Anomaly Score

        +

Explanation Reliability Score

        ↓

Threat Risk Score

        ↓

Threat Level
```


Threat levels:

| Risk Level | Meaning |
|---|---|
| Low | Normal monitoring |
| Medium | Suspicious activity |
| High | Critical threat |


## 10. ERGAR Adaptive Response Engine

ERGAR converts threat assessment results into automated security actions.

| Threat Level | Response Action |
|---|---|
| High | Block and Isolate |
| Medium | Monitor and Log |
| Low | Allow |


# Datasets


## CICIoT2023

CICIoT2023 is the primary dataset used for:

- Model training
- Validation
- Testing
- Streaming simulation
- Drift analysis


## Edge-IIoTset

Edge-IIoTset is used for:

- Cross dataset validation
- Generalization analysis


## Dataset Availability

Datasets are not included in this repository because of their large storage requirements.

Download the datasets separately and place them according to the configured paths.


# Project Structure


```
AdaptiveExplain-IDS/

│
├── src/
│   ├── models/
│   ├── explainability/
│   ├── drift/
│   ├── risk/
│   └── response/
│
├── scripts/
│
├── models/
│
├── results/
│
├── figures/
│
├── data/
│
├── config.yaml
│
├── main.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```


# Installation


Clone repository:

```bash
git clone <repository-url>

cd AdaptiveExplain-IDS
```


Create virtual environment:

```bash
python -m venv .venv
```


Activate environment:

Windows:

```bash
.venv\Scripts\activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```


# Configuration

The project configuration is managed using:

```
config.yaml
```

The configuration includes:

- Dataset paths
- Training parameters
- XGBoost settings
- Drift detection settings
- ERS parameters
- Response thresholds


# Execution Pipeline


## XGBoost Training

```bash
python scripts/09_train_xgboost.py
```


## Probability Calibration

```bash
python scripts/10_calibrate_xgboost.py
```


## Unknown Attack Detection

```bash
python scripts/12_unseen_attack_test.py
```


## Streaming Simulation

```bash
python scripts/14_streaming_simulation.py
```


## ADWIN Drift Detection

```bash
python scripts/15_adwin_drift_detection.py
```


## Adaptive Retraining

```bash
python scripts/16_adaptive_retraining.py
```


## SHAP Explainability

```bash
python scripts/17_shap_analysis.py
```


## Explanation Reliability Score

```bash
python scripts/18_ers_calculation.py
```


## Risk Engine

```bash
python scripts/19_risk_engine.py
```


## ERGAR Response

```bash
python scripts/20_ergar_response.py
```


## Edge-IIoTset Validation

```bash
python scripts/21_edge_validation.py
```


# Results

The framework generates the following outputs:


## Model Results

Contains:

- Training metrics
- Evaluation metrics
- Feature importance


## Calibration Results

Contains:

- Platt scaling results
- Isotonic calibration results


## SHAP Results

Contains:

- Feature importance
- Explanation analysis


## ERS Results

Contains:

- Explanation reliability score
- Explanation quality measurements


## Drift Results

Contains:

- Drift detection events
- ADWIN reports


## Risk and Response Results

Contains:

- Threat risk scores
- Threat levels
- Automated response actions


# Figures

Publication-ready figures are stored in:

```
figures/
```

Examples:

- Framework architecture diagram
- Model comparison chart
- Confusion matrix
- SHAP feature importance
- Drift detection visualization
- Adaptive retraining comparison
- Risk distribution
- Response distribution


# Research Focus

This project focuses on:

- IoT intrusion detection
- Explainable Artificial Intelligence
- Adaptive machine learning
- Concept drift detection
- Reliable threat explanation
- Automated cybersecurity response


# Limitations

- Large datasets are not included because of storage limitations.
- Cross-dataset evaluation may be affected by domain differences.
- Real-world deployment requires integration with live network monitoring infrastructure.


# Future Work

Future improvements include:

- Online learning algorithms
- Advanced domain adaptation methods
- Real-time packet-level deployment
- More diverse IoT datasets
- Automated response policy optimization


# License

This project is developed for academic and research purposes.