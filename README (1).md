# SC-DATransformer: AI-Powered Sociocultural Impact Analysis of Global Sporting Events

This repository provides the complete implementation, experimental pipeline, and explainable analytics framework for **SC-DATransformer**, a Sociocultural Dual-Attention TabTransformer designed to quantify and interpret the social impact of global sporting events using AI.

---

## Overview
Global sporting events influence culture, public sentiment, national identity, and economic perception. Traditional statistical models struggle to capture nonlinear, high-dimensional sociocultural interactions. This project introduces an interpretable dual-attention transformer architecture to address these limitations.

---

## Dataset Description
- ~70,000 event-level observations (2000–2024)
- 21 sociocultural, economic, and engagement variables
- Events include Olympics, FIFA World Cup, Asian Games, Commonwealth Games
- Labels are continuous **Social Impact Scores** derived via normalized aggregation, cross-source validation, and temporal smoothing

Dataset source:  
https://github.com/VisionLangAI/Sports-Event-Analysis

---

## Methodology
- Robust preprocessing and normalization
- Feature selection using Mutual Information, PCA, t-SNE, SHAP-RFE
- Dual-attention TabTransformer (SC-DATransformer)
- Contrastive calibration loss and monotonic constraints
- Extensive statistical validation and explainability (SHAP)

---

## Models Compared
- ElasticNet
- kNN
- LightGBM
- Proposed SC-DATransformer

---

## Evaluation Metrics
- MAE, RMSE, R²
- MAPE, sMAPE, MedAE
- Spearman’s ρ
- 95% Confidence Intervals (RMSE)
- Diebold–Mariano significance test

---

## Explainability
- Global and local SHAP analysis
- Feature importance ranking
- SHAP decision plots for individual predictions
- Calibration reliability and confidence stabilization curves

---

## Reproducibility
- Fixed random seed
- Explicit hyperparameter reporting
- Early stopping and regularization
- Publicly available dataset
- Deterministic data splits

---

## Ethical Considerations
- No personal or identifiable data
- Aggregated sociocultural indicators only
- Intended for decision support, not automated policy enforcement

---

## How to Run
```bash
pip install numpy pandas scikit-learn matplotlib seaborn lightgbm shap torch
python sc_dat_transformer_full_plots.py
```

---

## Citation
Deep Learning for Sports Events Analysis: Explainable Dual-Attention Transformer for Sociocultural Impact Prediction
