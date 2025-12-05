import pandas as pd
from Utilities import helper
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize helper
h = helper()

# Load datasets
symptoms_df = pd.read_csv("Disease and symptoms dataset.csv")
h.load_medications("medications.csv")
h.load_clinic_inventory("clinic_inventory_modified.csv")

print("=" * 80)
print("MODEL COMPARISON: Random Forest vs Naive Bayes")
print("=" * 80)

# Train Random Forest model
print("\n" + "=" * 80)
print("RANDOM FOREST MODEL")
print("=" * 80)
rf_model, rf_encoder, rf_train_acc, rf_test_acc, rf_X_train, rf_y_train, rf_X_test, rf_y_test = h.train_disease_model(symptoms_df, model_type='random_forest')
print(f"\nRandom Forest Training Accuracy: {rf_train_acc:.4f} ({rf_train_acc*100:.2f}%)")
print(f"Random Forest Testing Accuracy: {rf_test_acc:.4f} ({rf_test_acc*100:.2f}%)")

rf_y_pred = rf_model.predict(rf_X_test)
rf_y_pred_proba = rf_model.predict_proba(rf_X_test)
rf_precision = precision_score(rf_y_test, rf_y_pred, average='weighted', zero_division=0)
rf_recall = recall_score(rf_y_test, rf_y_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(rf_y_test, rf_y_pred, average='weighted', zero_division=0)

# Calculate ROC-AUC (multi-class: ovr = one-vs-rest)
# Use labels parameter to ensure consistency
rf_classes = list(range(len(rf_encoder.classes_)))
rf_roc_auc = roc_auc_score(rf_y_test, rf_y_pred_proba, multi_class='ovr', average='weighted', labels=rf_classes)

print(f"Precision: {rf_precision:.4f} ({rf_precision*100:.2f}%)")
print(f"Recall: {rf_recall:.4f} ({rf_recall*100:.2f}%)")
print(f"F1-Score: {rf_f1:.4f} ({rf_f1*100:.2f}%)")
print(f"ROC-AUC Score: {rf_roc_auc:.4f} ({rf_roc_auc*100:.2f}%)")

print("\nClassification Report:")
# Get unique classes in test set for proper reporting
unique_test_classes = sorted(set(rf_y_test))
test_class_names = rf_encoder.inverse_transform(unique_test_classes)
print(classification_report(rf_y_test, rf_y_pred, labels=unique_test_classes, target_names=test_class_names, zero_division=0))

# Train Naive Bayes model
print("\n" + "=" * 80)
print("NAIVE BAYES MODEL")
print("=" * 80)
nb_model, nb_encoder, nb_train_acc, nb_test_acc, nb_X_train, nb_y_train, nb_X_test, nb_y_test = h.train_disease_model(symptoms_df, model_type='naive_bayes')
print(f"\nNaive Bayes Training Accuracy: {nb_train_acc:.4f} ({nb_train_acc*100:.2f}%)")
print(f"Naive Bayes Testing Accuracy: {nb_test_acc:.4f} ({nb_test_acc*100:.2f}%)")

nb_y_pred = nb_model.predict(nb_X_test)
nb_y_pred_proba = nb_model.predict_proba(nb_X_test)
nb_precision = precision_score(nb_y_test, nb_y_pred, average='weighted', zero_division=0)
nb_recall = recall_score(nb_y_test, nb_y_pred, average='weighted', zero_division=0)
nb_f1 = f1_score(nb_y_test, nb_y_pred, average='weighted', zero_division=0)

# Calculate ROC-AUC (multi-class: ovr = one-vs-rest)
# Use labels parameter to ensure consistency
nb_classes = list(range(len(nb_encoder.classes_)))
nb_roc_auc = roc_auc_score(nb_y_test, nb_y_pred_proba, multi_class='ovr', average='weighted', labels=nb_classes)

print(f"Precision: {nb_precision:.4f} ({nb_precision*100:.2f}%)")
print(f"Recall: {nb_recall:.4f} ({nb_recall*100:.2f}%)")
print(f"F1-Score: {nb_f1:.4f} ({nb_f1*100:.2f}%)")
print(f"ROC-AUC Score: {nb_roc_auc:.4f} ({nb_roc_auc*100:.2f}%)")

print("\nClassification Report:")
# Get unique classes in test set for proper reporting
unique_test_classes_nb = sorted(set(nb_y_test))
test_class_names_nb = nb_encoder.inverse_transform(unique_test_classes_nb)
print(classification_report(nb_y_test, nb_y_pred, labels=unique_test_classes_nb, target_names=test_class_names_nb, zero_division=0))

# Store metrics in a DataFrame
metrics_data = {
    'Model': ['Random Forest', 'Naive Bayes'],
    'Training Accuracy': [rf_train_acc, nb_train_acc],
    'Testing Accuracy': [rf_test_acc, nb_test_acc],
    'Precision': [rf_precision, nb_precision],
    'Recall': [rf_recall, nb_recall],
    'F1-Score': [rf_f1, nb_f1],
    'ROC-AUC': [rf_roc_auc, nb_roc_auc]
}
metrics_df = pd.DataFrame(metrics_data)

# Save metrics to CSV
metrics_df.to_csv('model_metrics_comparison.csv', index=False)
print("\n✓ Metrics saved to 'model_metrics_comparison.csv'")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print("\n" + metrics_df.to_string(index=False))

print(f"\nTesting Accuracy Difference: {abs(rf_test_acc - nb_test_acc):.4f} ({abs(rf_test_acc - nb_test_acc)*100:.2f}%)")

if rf_test_acc > nb_test_acc:
    print(f"✓ Random Forest performs better by {(rf_test_acc - nb_test_acc)*100:.2f}%")
elif nb_test_acc > rf_test_acc:
    print(f"✓ Naive Bayes performs better by {(nb_test_acc - rf_test_acc)*100:.2f}%")
else:
    print("✓ Both models perform equally well")

# Test with example user input on both models
print("\n" + "=" * 80)
print("PREDICTION COMPARISON ON SAMPLE INPUT")
print("=" * 80)

user_input = {
    "Symptom_1": "back pain",
    "Symptom_2": "dizziness",
    "Symptom_3": "fatigue",
    "Symptom_4": "neck pain",
    "Symptom_5": "nausea",
    "Symptom_6": "headache",
    "Symptom_7": "blurred vision",
    "Symptom_8": "vomiting"
}

print(f"\nUser Symptoms: {', '.join(user_input.values())}")

print("\n--- Random Forest Predictions ---")
rf_results = h.predict_disease_probabilities(user_input, rf_model, rf_encoder, top_n=3, threshold=0.0)
if rf_results:
    for i, r in enumerate(rf_results, 1):
        print(f"\n{i}. Disease: {r['disease']} (Probability: {r['probability']:.3f})")
        print(f"   Matched Symptoms: {', '.join(r['matched_symptoms']) if r['matched_symptoms'] else 'None'}")
        print(f"   Medications: {', '.join(r['medications'][:3]) if r['medications'] else 'None'}")
else:
    print("No predictions generated")

print("\n--- Naive Bayes Predictions ---")
nb_results = h.predict_disease_probabilities(user_input, nb_model, nb_encoder, top_n=3, threshold=0.0)
if nb_results:
    for i, r in enumerate(nb_results, 1):
        print(f"\n{i}. Disease: {r['disease']} (Probability: {r['probability']:.3f})")
        print(f"   Matched Symptoms: {', '.join(r['matched_symptoms']) if r['matched_symptoms'] else 'None'}")
        print(f"   Medications: {', '.join(r['medications'][:3]) if r['medications'] else 'None'}")
else:
    print("No predictions generated")

# Create visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# 1. Bar chart comparing metrics
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_df.columns[1:]))
width = 0.35

bars1 = ax.bar(x - width/2, metrics_df.iloc[0, 1:], width, label='Random Forest', color='#2ecc71')
bars2 = ax.bar(x + width/2, metrics_df.iloc[1, 1:], width, label='Naive Bayes', color='#3498db')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.columns[1:])
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_metrics_comparison.png")
plt.close()

# 2. ROC Curve Comparison with Micro-Average
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the output for ROC curve calculation
n_classes = len(rf_encoder.classes_)
rf_y_test_bin = label_binarize(rf_y_test, classes=list(range(n_classes)))
nb_y_test_bin = label_binarize(nb_y_test, classes=list(range(n_classes)))

# Compute micro-average ROC curve and ROC area
rf_fpr, rf_tpr, _ = roc_curve(rf_y_test_bin.ravel(), rf_y_pred_proba.ravel())
rf_roc_auc_micro = auc(rf_fpr, rf_tpr)

nb_fpr, nb_tpr, _ = roc_curve(nb_y_test_bin.ravel(), nb_y_pred_proba.ravel())
nb_roc_auc_micro = auc(nb_fpr, nb_tpr)

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(rf_fpr, rf_tpr, color='#2ecc71', lw=2.5, 
        label=f'Random Forest (AUC = {rf_roc_auc_micro:.4f})')
ax.plot(nb_fpr, nb_tpr, color='#3498db', lw=2.5, 
        label=f'Naive Bayes (AUC = {nb_roc_auc_micro:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve Comparison (Micro-Average)', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curve_comparison.png")
plt.close()

print("\n" + "=" * 80)
