This README is a general overview of our group 8 final project in ENGG 680. Group members are: Caydin Cleland:30162848, Chethan Karthikeya Sundharesh: 30273252 Sepehr Akhiani: 30265830 Ramtin Chelongarian: 30263429
#System
A machine learning-based disease prediction system that uses patient symptoms to predict potential diseases and recommend nearby clinics with appropriate medications.

## Overview

This project implements and compares two machine learning models (Random Forest and Naive Bayes) for disease prediction based on symptom data. The system processes 377 symptoms across 754 disease classes and provides top-K predictions with confidence scores.

## Dataset

**Disease and symptoms dataset.csv** (182 MB - not included in repository)
- 246,945 samples
- 377 symptom features
- 754 disease classes
- Multi-hot encoded symptom representation

*Note: Download the dataset separately and place it in the Project folder before running the code.*
Link to  data set: https://data.mendeley.com/datasets/2cxccsxydc/1

## Model Performance

### Random Forest (25 trees)
- Test Accuracy: 83.71%
- ROC-AUC (Weighted): 97.89%
- Top-3 Accuracy: 93.83%
- Top-5 Accuracy: 95.38%

### Naive Bayes (MultinomialNB)
- Test Accuracy: 83.89%
- ROC-AUC (Weighted): 99.97%
- Top-3 Accuracy: 94.42%
- Top-5 Accuracy: 96.83%

## Files

### Core Scripts
- **main.py** - Main application for disease prediction with clinic recommendations
- **Utilities.py** - Utility functions for data processing, multi-hot encoding, and OSRM distance calculation
- **model_comparison.py** - Compares Random Forest and Naive Bayes models with ROC curves
- **train_and_save_model.py** - Trains and saves the machine learning model
- **test_new_dataset.py** - Tests model on new dataset configurations

### Visualization Scripts
- **generate_additional_visualizations.py** - Creates feature importance, precision-recall curves, and F1 histograms
- **extract_detailed_metrics.py** - Generates detailed performance metrics with macro/weighted averages

### Data Files
- **clinic_inventory.csv** - Clinic location and medication inventory data
- **clinic_inventory_modified.csv** - Modified clinic inventory
- **medications.csv** - Disease-medication mapping

### Model Files
- **trained_model.joblib** (130 MB - not included in repository) - Pre-trained Random Forest model

### Metrics and Visualizations Folder
*Located in "Model Metrics and model comparison data" subfolder:*
- ROC curves for both models
- Confusion matrices (CSV and PNG)
- Feature importance plots and data
- Precision-recall curves
- F1 score histograms
- Class frequency distributions
- Detailed metrics comparison (CSV)

## Key Features

1. **Multi-hot Symptom Encoding** - Converts symptom strings into binary feature vectors
2. **Top-K Predictions** - Provides multiple disease predictions with confidence scores
3. **Clinic Recommendations** - Uses OSRM API to find nearest clinics with required medications
4. **Road Distance Calculation** - Real-world driving distances instead of Euclidean distance
5. **Comprehensive Evaluation** - Multiple metrics including ROC-AUC, precision, recall, F1-score

## Model Configuration

### Random Forest
```python
RandomForestClassifier(n_estimators=25, random_state=42)
# All other parameters at default values
