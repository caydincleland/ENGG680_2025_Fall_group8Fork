import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utilities import helper
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Initialize helper
h = helper()

# Load dataset
print("Loading dataset...")
symptoms_df = pd.read_csv("Disease and symptoms dataset.csv")

print("\n" + "=" * 80)
print("GENERATING ADDITIONAL VISUALIZATIONS AND DATA")
print("=" * 80)

# ============================================================================
# 1. DISTRIBUTION OF CLASS FREQUENCIES
# ============================================================================
print("\n[1/6] Creating class frequency distribution...")
disease_col = symptoms_df.columns[0]
class_counts = symptoms_df[disease_col].value_counts().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(class_counts.values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Number of Samples per Disease', fontsize=12)
plt.ylabel('Frequency (Number of Diseases)', fontsize=12)
plt.title('Distribution of Class Frequencies', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(class_counts.values, vert=True)
plt.ylabel('Number of Samples per Disease', fontsize=12)
plt.title('Class Frequency Boxplot', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('class_frequency_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: class_frequency_distribution.png")

# Save class frequency data to CSV
class_freq_df = pd.DataFrame({
    'Disease': class_counts.index,
    'Sample_Count': class_counts.values
})
class_freq_df.to_csv('class_frequency_data.csv', index=False)
print("✓ Saved: class_frequency_data.csv")

# Print statistics
print(f"\nClass Frequency Statistics:")
print(f"  Total diseases: {len(class_counts)}")
print(f"  Mean samples per disease: {class_counts.mean():.2f}")
print(f"  Median samples per disease: {class_counts.median():.2f}")
print(f"  Min samples: {class_counts.min()}")
print(f"  Max samples: {class_counts.max()}")
print(f"  Std deviation: {class_counts.std():.2f}")

# ============================================================================
# 2. TRAIN MODELS FOR REMAINING VISUALIZATIONS
# ============================================================================
print("\n[2/6] Training Random Forest model...")
rf_model, rf_encoder, rf_train_acc, rf_test_acc, rf_X_train, rf_y_train, rf_X_test, rf_y_test = h.train_disease_model(
    symptoms_df, model_type='random_forest'
)
rf_y_pred = rf_model.predict(rf_X_test)
rf_y_pred_proba = rf_model.predict_proba(rf_X_test)

print("\n[3/6] Training Naive Bayes model...")
nb_model, nb_encoder, nb_train_acc, nb_test_acc, nb_X_train, nb_y_train, nb_X_test, nb_y_test = h.train_disease_model(
    symptoms_df, model_type='naive_bayes'
)
nb_y_pred = nb_model.predict(nb_X_test)
nb_y_pred_proba = nb_model.predict_proba(nb_X_test)

# ============================================================================
# 3. F1-SCORE PER CLASS HISTOGRAM
# ============================================================================
print("\n[4/6] Creating F1-score per class histograms...")
from sklearn.metrics import f1_score

# Calculate per-class F1 scores
rf_f1_per_class = []
nb_f1_per_class = []

unique_classes = sorted(set(rf_y_test))
for cls in unique_classes:
    # Random Forest
    y_true_binary = (rf_y_test == cls).astype(int)
    y_pred_binary = (rf_y_pred == cls).astype(int)
    if y_true_binary.sum() > 0:  # Only calculate if class exists in test set
        rf_f1_per_class.append(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    
    # Naive Bayes
    y_true_binary_nb = (nb_y_test == cls).astype(int)
    y_pred_binary_nb = (nb_y_pred == cls).astype(int)
    if y_true_binary_nb.sum() > 0:
        nb_f1_per_class.append(f1_score(y_true_binary_nb, y_pred_binary_nb, zero_division=0))

# Plot histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(rf_f1_per_class, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
axes[0].axvline(np.mean(rf_f1_per_class), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rf_f1_per_class):.3f}')
axes[0].set_xlabel('F1-Score', fontsize=12)
axes[0].set_ylabel('Number of Classes', fontsize=12)
axes[0].set_title('Random Forest: F1-Score Distribution per Class', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(nb_f1_per_class, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(np.mean(nb_f1_per_class), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(nb_f1_per_class):.3f}')
axes[1].set_xlabel('F1-Score', fontsize=12)
axes[1].set_ylabel('Number of Classes', fontsize=12)
axes[1].set_title('Naive Bayes: F1-Score Distribution per Class', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('f1_score_per_class_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Saved: f1_score_per_class_histogram.png")

# ============================================================================
# 4. FEATURE IMPORTANCE PLOT (Random Forest only)
# ============================================================================
print("\n[5/6] Creating feature importance plot...")
feature_importances = rf_model.feature_importances_
feature_names = h.all_symptoms

# Get top 30 most important features
top_n_features = 30
indices = np.argsort(feature_importances)[-top_n_features:]
top_features = [feature_names[i] for i in indices]
top_importances = feature_importances[indices]

plt.figure(figsize=(10, 12))
plt.barh(range(len(top_importances)), top_importances, color='teal', edgecolor='black')
plt.yticks(range(len(top_importances)), top_features, fontsize=9)
plt.xlabel('Feature Importance', fontsize=12)
plt.title(f'Top {top_n_features} Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance_plot.png")

# Save all feature importances to CSV
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv('feature_importance_data.csv', index=False)
print("✓ Saved: feature_importance_data.csv")

# ============================================================================
# 5. CONFUSION MATRIX DATA TO CSV
# ============================================================================
print("\n[6/6] Generating confusion matrix data...")

# Random Forest confusion matrix
rf_cm = confusion_matrix(rf_y_test, rf_y_pred, labels=unique_classes)
rf_class_names = rf_encoder.inverse_transform(unique_classes)
rf_cm_df = pd.DataFrame(rf_cm, index=rf_class_names, columns=rf_class_names)
rf_cm_df.to_csv('confusion_matrix_random_forest.csv')
print("✓ Saved: confusion_matrix_random_forest.csv")

# Naive Bayes confusion matrix
nb_cm = confusion_matrix(nb_y_test, nb_y_pred, labels=unique_classes)
nb_class_names = nb_encoder.inverse_transform(unique_classes)
nb_cm_df = pd.DataFrame(nb_cm, index=nb_class_names, columns=nb_class_names)
nb_cm_df.to_csv('confusion_matrix_naive_bayes.csv')
print("✓ Saved: confusion_matrix_naive_bayes.csv")

# ============================================================================
# 6. PRECISION-RECALL CURVES
# ============================================================================
print("\n[7/7] Creating precision-recall curves...")

# Binarize labels for multi-class precision-recall
rf_classes = list(range(len(rf_encoder.classes_)))
y_test_binarized = label_binarize(rf_y_test, classes=rf_classes)

# Calculate micro-average precision-recall curve for Random Forest
precision_rf_micro = dict()
recall_rf_micro = dict()
precision_rf_micro["micro"], recall_rf_micro["micro"], _ = precision_recall_curve(
    y_test_binarized.ravel(), rf_y_pred_proba.ravel()
)
rf_pr_auc = auc(recall_rf_micro["micro"], precision_rf_micro["micro"])

# Calculate micro-average precision-recall curve for Naive Bayes
y_test_binarized_nb = label_binarize(nb_y_test, classes=rf_classes)
precision_nb_micro = dict()
recall_nb_micro = dict()
precision_nb_micro["micro"], recall_nb_micro["micro"], _ = precision_recall_curve(
    y_test_binarized_nb.ravel(), nb_y_pred_proba.ravel()
)
nb_pr_auc = auc(recall_nb_micro["micro"], precision_nb_micro["micro"])

# Plot precision-recall curves
plt.figure(figsize=(10, 7))
plt.plot(recall_rf_micro["micro"], precision_rf_micro["micro"], 
         label=f'Random Forest (AUC = {rf_pr_auc:.4f})', linewidth=2, color='forestgreen')
plt.plot(recall_nb_micro["micro"], precision_nb_micro["micro"], 
         label=f'Naive Bayes (AUC = {nb_pr_auc:.4f})', linewidth=2, color='coral')
plt.plot([0, 1], [1, 0], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.title('Precision-Recall Curve Comparison (Micro-Average)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curve.png")

# Save PR curve data to CSV
pr_curve_data = pd.DataFrame({
    'RF_Recall': recall_rf_micro["micro"],
    'RF_Precision': precision_rf_micro["micro"],
    'NB_Recall': recall_nb_micro["micro"][:len(recall_rf_micro["micro"])],
    'NB_Precision': precision_nb_micro["micro"][:len(precision_rf_micro["micro"])]
})
pr_curve_data.to_csv('precision_recall_curve_data.csv', index=False)
print("✓ Saved: precision_recall_curve_data.csv")

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS AND DATA FILES GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated files:")
print("  1. class_frequency_distribution.png")
print("  2. class_frequency_data.csv")
print("  3. f1_score_per_class_histogram.png")
print("  4. feature_importance_plot.png")
print("  5. feature_importance_data.csv")
print("  6. confusion_matrix_random_forest.csv")
print("  7. confusion_matrix_naive_bayes.csv")
print("  8. precision_recall_curve.png")
print("  9. precision_recall_curve_data.csv")
