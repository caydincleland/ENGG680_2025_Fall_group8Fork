import pandas as pd
from Utilities import helper
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize helper
h = helper()

# Load datasets
symptoms_df = pd.read_csv("symptoms_df.csv")
h.load_medications("medications.csv")
h.load_clinic_inventory("clinic_inventory_modified.csv")

print("=" * 80)
print("MODEL COMPARISON: Random Forest vs Naive Bayes")
print("=" * 80)

# Train Random Forest model
print("\n" + "=" * 80)
print("RANDOM FOREST MODEL")
print("=" * 80)
rf_model, rf_encoder, rf_accuracy, rf_X_test, rf_y_test = h.train_disease_model(symptoms_df, model_type='random_forest')
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

rf_y_pred = rf_model.predict(rf_X_test)
rf_precision = precision_score(rf_y_test, rf_y_pred, average='weighted', zero_division=0)
rf_recall = recall_score(rf_y_test, rf_y_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(rf_y_test, rf_y_pred, average='weighted', zero_division=0)

print(f"Precision: {rf_precision:.4f} ({rf_precision*100:.2f}%)")
print(f"Recall: {rf_recall:.4f} ({rf_recall*100:.2f}%)")
print(f"F1-Score: {rf_f1:.4f} ({rf_f1*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(rf_y_test, rf_y_pred, target_names=rf_encoder.classes_, zero_division=0))

# Train Naive Bayes model
print("\n" + "=" * 80)
print("NAIVE BAYES MODEL")
print("=" * 80)
nb_model, nb_encoder, nb_accuracy, nb_X_test, nb_y_test = h.train_disease_model(symptoms_df, model_type='naive_bayes')
print(f"\nNaive Bayes Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")

nb_y_pred = nb_model.predict(nb_X_test)
nb_precision = precision_score(nb_y_test, nb_y_pred, average='weighted', zero_division=0)
nb_recall = recall_score(nb_y_test, nb_y_pred, average='weighted', zero_division=0)
nb_f1 = f1_score(nb_y_test, nb_y_pred, average='weighted', zero_division=0)

print(f"Precision: {nb_precision:.4f} ({nb_precision*100:.2f}%)")
print(f"Recall: {nb_recall:.4f} ({nb_recall*100:.2f}%)")
print(f"F1-Score: {nb_f1:.4f} ({nb_f1*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(nb_y_test, nb_y_pred, target_names=nb_encoder.classes_, zero_division=0))

# Store metrics in a DataFrame
metrics_data = {
    'Model': ['Random Forest', 'Naive Bayes'],
    'Accuracy': [rf_accuracy, nb_accuracy],
    'Precision': [rf_precision, nb_precision],
    'Recall': [rf_recall, nb_recall],
    'F1-Score': [rf_f1, nb_f1]
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

print(f"\nAccuracy Difference: {abs(rf_accuracy - nb_accuracy):.4f} ({abs(rf_accuracy - nb_accuracy)*100:.2f}%)")

if rf_accuracy > nb_accuracy:
    print(f"✓ Random Forest performs better by {(rf_accuracy - nb_accuracy)*100:.2f}%")
elif nb_accuracy > rf_accuracy:
    print(f"✓ Naive Bayes performs better by {(nb_accuracy - rf_accuracy)*100:.2f}%")
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
    "Symptom_4": "neck pain"
}

print(f"\nUser Symptoms: {', '.join(user_input.values())}")

print("\n--- Random Forest Predictions ---")
rf_results = h.predict_disease_probabilities(user_input, rf_model, rf_encoder, top_n=3)
for i, r in enumerate(rf_results, 1):
    print(f"\n{i}. Disease: {r['disease']} (Probability: {r['probability']:.3f})")
    print(f"   Matched Symptoms: {', '.join(r['matched_symptoms']) if r['matched_symptoms'] else 'None'}")
    print(f"   Medications: {', '.join(r['medications'][:3]) if r['medications'] else 'None'}")

print("\n--- Naive Bayes Predictions ---")
nb_results = h.predict_disease_probabilities(user_input, nb_model, nb_encoder, top_n=3)
for i, r in enumerate(nb_results, 1):
    print(f"\n{i}. Disease: {r['disease']} (Probability: {r['probability']:.3f})")
    print(f"   Matched Symptoms: {', '.join(r['matched_symptoms']) if r['matched_symptoms'] else 'None'}")
    print(f"   Medications: {', '.join(r['medications'][:3]) if r['medications'] else 'None'}")

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
ax.set_ylim([0.98, 1.0])
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

# 2. Confusion Matrix for Random Forest
rf_cm = confusion_matrix(rf_y_test, rf_y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_encoder.classes_, 
            yticklabels=rf_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Random Forest - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix_random_forest.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_random_forest.png")
plt.close()

# 3. Confusion Matrix for Naive Bayes
nb_cm = confusion_matrix(nb_y_test, nb_y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=nb_encoder.classes_,
            yticklabels=nb_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Naive Bayes - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('confusion_matrix_naive_bayes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_naive_bayes.png")
plt.close()

# 4. Side-by-side confusion matrix comparison (normalized)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Normalize confusion matrices
rf_cm_norm = rf_cm.astype('float') / rf_cm.sum(axis=1)[:, np.newaxis]
nb_cm_norm = nb_cm.astype('float') / nb_cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(rf_cm_norm, annot=False, fmt='.2f', cmap='Blues', ax=axes[0],
            xticklabels=rf_encoder.classes_,
            yticklabels=rf_encoder.classes_,
            cbar_kws={'label': 'Proportion'})
axes[0].set_title('Random Forest (Normalized)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=90, labelsize=7)
axes[0].tick_params(axis='y', rotation=0, labelsize=7)

sns.heatmap(nb_cm_norm, annot=False, fmt='.2f', cmap='Greens', ax=axes[1],
            xticklabels=nb_encoder.classes_,
            yticklabels=nb_encoder.classes_,
            cbar_kws={'label': 'Proportion'})
axes[1].set_title('Naive Bayes (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=90, labelsize=7)
axes[1].tick_params(axis='y', rotation=0, labelsize=7)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix_comparison.png")
plt.close()

print("\n" + "=" * 80)
