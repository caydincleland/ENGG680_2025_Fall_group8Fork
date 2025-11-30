import pandas as pd
from Utilities import helper
# Initialize helper
h = helper()

# Load datasets
symptoms_df = pd.read_csv("symptoms_df.csv")
h.load_medications("medications.csv")
h.load_clinic_inventory("clinic_inventory_modified.csv")

# Train model
disease_model, disease_encoder, accuracy = h.train_disease_model(symptoms_df)
print(f"Model accuracy: {accuracy:.4f}")

# Example user input
user_input = {
    "Symptom_1": "back pain",
    "Symptom_2": "dizziness",
    "Symptom_3": "fatigue",
    "Symptom_4": "neck pain"
}

# Predict probable diseases
results = h.predict_disease_probabilities(user_input, disease_model, disease_encoder, top_n=3)

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


