import pandas as pd
import os
from Utilities import helper

# Change to script directory to ensure relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize helper
h = helper()
# Load datasets
h.load_medications("medications.csv")
h.load_clinic_inventory("clinic_inventory_modified.csv")

# Load pre-trained model (or train if model doesn't exist)
model_file = "trained_model.joblib"

if os.path.exists(model_file):
    print("Loading pre-trained model...")
    disease_model, disease_encoder = h.load_model(model_file)
else:
    print("No saved model found. Training new model...")
    symptoms_df = pd.read_csv("Disease and symptoms dataset.csv")
    disease_model, disease_encoder, train_accuracy, test_accuracy, X_train, y_train, X_test, y_test = h.train_disease_model(symptoms_df)
    print(f"Model training accuracy: {train_accuracy:.4f}")
    print(f"Model testing accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    h.save_model(disease_model, disease_encoder, model_file)

# Example user input
user_input = {
    "Symptom_1": "abnormal appearing skin",
    "Symptom_2": "acne or pimples",
    "Symptom_3": "skin lesion",
    "Symptom_4": "skin swelling",
    "Symptom_5": "skin rash",
    "Symptom_6": "itching of skin",
    "Symptom_7": "itchy scalp",
    
    
    
}

# Predict probable diseases (lowered threshold to 0.1 to see more results)
results = h.predict_disease_probabilities(user_input, disease_model, disease_encoder, top_n=3, threshold=0.1)

# Print disease predictions and medications
for r in results:
    print(f"\nDisease: {r['disease']} (Probability: {r['probability']})")
    print("Common Symptoms:", ", ".join(r['common_symptoms']))
    print("Matched Symptoms:", ", ".join(r['matched_symptoms']) if r['matched_symptoms'] else "None")
    print("Unmatched Symptoms:", ", ".join(r['unmatched_symptoms']) if r['unmatched_symptoms'] else "None")
    print("Suggested Medications:", ", ".join(r['medications']) if r['medications'] else "None, consult doctor")

# Find clinics with medications
clinics_for_diseases = h.find_clinics_for_predicted_diseases(results)
for disease, clinics in clinics_for_diseases.items():
    print(f"\nClinics for {disease}:")
    for c in clinics:
        print(f" - {c['clinic_name']} (Available: {', '.join(c['available_medicines'])})")

top_clinics = h.get_top_clinics_by_road_distance(
    user_lat=51.045,
    user_lon=-114.063,
    clinics=clinics_for_diseases
)

print("Top 3 nearest clinics:")
for c in top_clinics:
    print(f" - {c['clinic_name']} (Distance: {c['distance_km']:.3f} km)")


