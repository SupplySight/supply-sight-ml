# Supply Chain Sleuth: ML Pipeline

This repository contains the Machine Learning components for Supply Chain Sleuth, an NLP-driven risk assessment tool. The system uses a fine-tuned DistilBERT model to evaluate ethical risks in corporate supply chains based on news headlines and ESG reports.

## Model Overview

The system utilizes a `distilbert-base-uncased` transformer model fine-tuned for a 3-class classification task. The goal is to identify and quantify ethical risk with a focus on high recall for labor and environmental violations.

- **Architecture:** DistilBERT (Bidirectional Encoder Representations from Transformers)
- **Task:** 3-Class Sentiment Classification (Risk Assessment)
- **Output:** Numerical Risk Score (0-100) and Status Label (RED, YELLOW, GREEN)

## Repository Structure

* `SupplyChainSleuth_ModelTraining.ipynb`: Main notebook for data tokenization, model fine-tuning, and evaluation.
* `train.py`: The training entry point for SageMaker GPU instances, implementing adversarial fine-tuning.
* `inference.py`: Custom inference handler for SageMaker Serverless Endpoint. It calculates the weighted 0-100 risk score.
* `SupplyChainSleuth_DataIngestion.ipynb`: Pipeline for fetching and cleaning ESG-related datasets from DynamoDB.

## Scoring Logic (Risk-Centric)

The system is configured so that a **higher percentage equals higher risk**. The `inference.py` script applies Softmax to the model logits and calculates a weighted average:

| Class | Label | Weight | Description |
| :--- | :--- | :--- | :--- |
| **RED** | 0 | 100 | Severe ethical violations (e.g., forced labor). |
| **YELLOW** | 1 | 50 | Moderate risk or supply chain disruptions. |
| **GREEN** | 2 | 0 | Low risk, sustainable practices, or neutral news. |

**Formula:** $RiskScore = (P_{Red} \times 100) + (P_{Yellow} \times 50) + (P_{Green} \times 0)$

## Deployment

The model is designed for deployment on **Amazon SageMaker Serverless Inference** to minimize costs.

1. **Packaging:** The fine-tuned model weights and `inference.py` are bundled into `model.tar.gz`.
2. **Inference:** The endpoint accepts a JSON payload containing a `headline` and returns a `risk_score` (0-100) and a `status` label.
