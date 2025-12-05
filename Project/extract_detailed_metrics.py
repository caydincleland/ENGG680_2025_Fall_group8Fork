import pandas as pd
from Utilities import helper
from sklearn.metrics import classification_report, top_k_accuracy_score, roc_auc_score
import numpy as np

# Initialize helper
h = helper()

# Load datasets
symptoms_df = pd.read_csv("Disease and symptoms dataset.csv")

print("=" * 80)
print("DETAILED MODEL PERFORMANCE METRICS")
print("=" * 80)

# Train Random Forest model
print("\n" + "=" * 80)
print("RANDOM FOREST MODEL")
print("=" * 80)
rf_model, rf_encoder, rf_train_acc, rf_test_acc, rf_X_train, rf_y_train, rf_X_test, rf_y_test = h.train_disease_model(symptoms_df, model_type='random_forest')

rf_y_pred = rf_model.predict(rf_X_test)
rf_y_pred_proba = rf_model.predict_proba(rf_X_test)

# Get classification report with output_dict=True to extract macro/weighted
rf_report = classification_report(rf_y_test, rf_y_pred, output_dict=True, zero_division=0)

# Define classes list for ROC-AUC and Top-K calculations
rf_classes = list(range(len(rf_encoder.classes_)))

# Calculate Top-K accuracy
rf_top1_acc = top_k_accuracy_score(rf_y_test, rf_y_pred_proba, k=1, labels=rf_classes)
rf_top3_acc = top_k_accuracy_score(rf_y_test, rf_y_pred_proba, k=3, labels=rf_classes)
rf_top5_acc = top_k_accuracy_score(rf_y_test, rf_y_pred_proba, k=5, labels=rf_classes)

# Calculate ROC-AUC macro
rf_roc_auc_macro = roc_auc_score(rf_y_test, rf_y_pred_proba, multi_class='ovr', average='macro', labels=rf_classes)
rf_roc_auc_weighted = roc_auc_score(rf_y_test, rf_y_pred_proba, multi_class='ovr', average='weighted', labels=rf_classes)

print(f"\nAccuracy (Test): {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")
print(f"\nPrecision:")
print(f"  - Macro: {rf_report['macro avg']['precision']:.4f} ({rf_report['macro avg']['precision']*100:.2f}%)")
print(f"  - Weighted: {rf_report['weighted avg']['precision']:.4f} ({rf_report['weighted avg']['precision']*100:.2f}%)")
print(f"\nRecall:")
print(f"  - Macro: {rf_report['macro avg']['recall']:.4f} ({rf_report['macro avg']['recall']*100:.2f}%)")
print(f"  - Weighted: {rf_report['weighted avg']['recall']:.4f} ({rf_report['weighted avg']['recall']*100:.2f}%)")
print(f"\nF1-Score:")
print(f"  - Macro: {rf_report['macro avg']['f1-score']:.4f} ({rf_report['macro avg']['f1-score']*100:.2f}%)")
print(f"  - Weighted: {rf_report['weighted avg']['f1-score']:.4f} ({rf_report['weighted avg']['f1-score']*100:.2f}%)")
print(f"\nROC-AUC:")
print(f"  - Macro: {rf_roc_auc_macro:.4f} ({rf_roc_auc_macro*100:.2f}%)")
print(f"  - Weighted: {rf_roc_auc_weighted:.4f} ({rf_roc_auc_weighted*100:.2f}%)")
print(f"\nTop-K Accuracy:")
print(f"  - Top-1: {rf_top1_acc:.4f} ({rf_top1_acc*100:.2f}%)")
print(f"  - Top-3: {rf_top3_acc:.4f} ({rf_top3_acc*100:.2f}%)")
print(f"  - Top-5: {rf_top5_acc:.4f} ({rf_top5_acc*100:.2f}%)")

# Train Naive Bayes model
print("\n" + "=" * 80)
print("NAIVE BAYES MODEL")
print("=" * 80)
nb_model, nb_encoder, nb_train_acc, nb_test_acc, nb_X_train, nb_y_train, nb_X_test, nb_y_test = h.train_disease_model(symptoms_df, model_type='naive_bayes')

nb_y_pred = nb_model.predict(nb_X_test)
nb_y_pred_proba = nb_model.predict_proba(nb_X_test)

# Get classification report with output_dict=True to extract macro/weighted
nb_report = classification_report(nb_y_test, nb_y_pred, output_dict=True, zero_division=0)

# Define classes list for ROC-AUC and Top-K calculations
nb_classes = list(range(len(nb_encoder.classes_)))

# Calculate Top-K accuracy
nb_top1_acc = top_k_accuracy_score(nb_y_test, nb_y_pred_proba, k=1, labels=nb_classes)
nb_top3_acc = top_k_accuracy_score(nb_y_test, nb_y_pred_proba, k=3, labels=nb_classes)
nb_top5_acc = top_k_accuracy_score(nb_y_test, nb_y_pred_proba, k=5, labels=nb_classes)

# Calculate ROC-AUC macro
nb_roc_auc_macro = roc_auc_score(nb_y_test, nb_y_pred_proba, multi_class='ovr', average='macro', labels=nb_classes)
nb_roc_auc_weighted = roc_auc_score(nb_y_test, nb_y_pred_proba, multi_class='ovr', average='weighted', labels=nb_classes)

print(f"\nAccuracy (Test): {nb_test_acc:.4f} ({nb_test_acc*100:.2f}%)")
print(f"\nPrecision:")
print(f"  - Macro: {nb_report['macro avg']['precision']:.4f} ({nb_report['macro avg']['precision']*100:.2f}%)")
print(f"  - Weighted: {nb_report['weighted avg']['precision']:.4f} ({nb_report['weighted avg']['precision']*100:.2f}%)")
print(f"\nRecall:")
print(f"  - Macro: {nb_report['macro avg']['recall']:.4f} ({nb_report['macro avg']['recall']*100:.2f}%)")
print(f"  - Weighted: {nb_report['weighted avg']['recall']:.4f} ({nb_report['weighted avg']['recall']*100:.2f}%)")
print(f"\nF1-Score:")
print(f"  - Macro: {nb_report['macro avg']['f1-score']:.4f} ({nb_report['macro avg']['f1-score']*100:.2f}%)")
print(f"  - Weighted: {nb_report['weighted avg']['f1-score']:.4f} ({nb_report['weighted avg']['f1-score']*100:.2f}%)")
print(f"\nROC-AUC:")
print(f"  - Macro: {nb_roc_auc_macro:.4f} ({nb_roc_auc_macro*100:.2f}%)")
print(f"  - Weighted: {nb_roc_auc_weighted:.4f} ({nb_roc_auc_weighted*100:.2f}%)")
print(f"\nTop-K Accuracy:")
print(f"  - Top-1: {nb_top1_acc:.4f} ({nb_top1_acc*100:.2f}%)")
print(f"  - Top-3: {nb_top3_acc:.4f} ({nb_top3_acc*100:.2f}%)")
print(f"  - Top-5: {nb_top5_acc:.4f} ({nb_top5_acc*100:.2f}%)")

# Create comprehensive comparison table
print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON TABLE")
print("=" * 80)

comparison_data = {
    'Metric': [
        'Test Accuracy',
        'Precision (Macro)',
        'Precision (Weighted)',
        'Recall (Macro)',
        'Recall (Weighted)',
        'F1-Score (Macro)',
        'F1-Score (Weighted)',
        'ROC-AUC (Macro)',
        'ROC-AUC (Weighted)',
        'Top-1 Accuracy',
        'Top-3 Accuracy',
        'Top-5 Accuracy'
    ],
    'Random Forest': [
        f"{rf_test_acc:.4f}",
        f"{rf_report['macro avg']['precision']:.4f}",
        f"{rf_report['weighted avg']['precision']:.4f}",
        f"{rf_report['macro avg']['recall']:.4f}",
        f"{rf_report['weighted avg']['recall']:.4f}",
        f"{rf_report['macro avg']['f1-score']:.4f}",
        f"{rf_report['weighted avg']['f1-score']:.4f}",
        f"{rf_roc_auc_macro:.4f}",
        f"{rf_roc_auc_weighted:.4f}",
        f"{rf_top1_acc:.4f}",
        f"{rf_top3_acc:.4f}",
        f"{rf_top5_acc:.4f}"
    ],
    'Naive Bayes': [
        f"{nb_test_acc:.4f}",
        f"{nb_report['macro avg']['precision']:.4f}",
        f"{nb_report['weighted avg']['precision']:.4f}",
        f"{nb_report['macro avg']['recall']:.4f}",
        f"{nb_report['weighted avg']['recall']:.4f}",
        f"{nb_report['macro avg']['f1-score']:.4f}",
        f"{nb_report['weighted avg']['f1-score']:.4f}",
        f"{nb_roc_auc_macro:.4f}",
        f"{nb_roc_auc_weighted:.4f}",
        f"{nb_top1_acc:.4f}",
        f"{nb_top3_acc:.4f}",
        f"{nb_top5_acc:.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('detailed_metrics_comparison.csv', index=False)
print("\n✓ Detailed metrics saved to 'detailed_metrics_comparison.csv'")
