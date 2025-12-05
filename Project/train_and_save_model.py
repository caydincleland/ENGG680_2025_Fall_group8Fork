"""
One-time script to train and save the Random Forest model.
Run this once, then use main.py which will load the saved model.
"""
import pandas as pd
from Utilities import helper

# Initialize helper
h = helper()

# Load symptom dataset
print("Loading dataset...")
symptoms_df = pd.read_csv("Disease and symptoms dataset.csv")
print(f"Dataset shape: {symptoms_df.shape}")

# Train model
print("\nTraining Random Forest model...")
disease_model, disease_encoder, train_accuracy, test_accuracy, X_train, y_train, X_test, y_test = h.train_disease_model(
    symptoms_df, 
    model_type='random_forest'
)

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Number of diseases: {len(disease_encoder.classes_)}")
print(f"Number of symptoms: {len(h.all_symptoms)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Save the model
print(f"\n{'='*60}")
model_file = "trained_model.joblib"
h.save_model(disease_model, disease_encoder, model_file)
print(f"{'='*60}")
print(f"\nModel saved successfully!")
print(f"You can now run main.py without retraining.")
print(f"The model will be loaded from '{model_file}'")
