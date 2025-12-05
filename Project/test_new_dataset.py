import pandas as pd
from Utilities import helper

print("Loading dataset...")
symptoms_df = pd.read_csv("Disease and symptoms dataset.csv")
print(f"Dataset loaded: {symptoms_df.shape[0]} rows, {symptoms_df.shape[1]} columns")
print(f"First column (diseases): {symptoms_df.columns[0]}")
print(f"Number of unique diseases: {symptoms_df[symptoms_df.columns[0]].nunique()}")
print(f"Sample diseases: {symptoms_df[symptoms_df.columns[0]].unique()[:5]}")

print("\nInitializing helper and training model...")
h = helper()
h.load_medications("medications.csv")

# Train with just 100 samples for quick test
test_df = symptoms_df.head(100)
print(f"Training on {len(test_df)} samples...")

model, encoder, train_acc, test_acc, X_train, y_train, X_test, y_test = h.train_disease_model(test_df, model_type='random_forest')
print(f"\n✓ Model trained successfully!")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")
print(f"Number of symptoms in model: {len(h.all_symptoms)}")
print(f"Sample symptoms: {h.all_symptoms[:10]}")
