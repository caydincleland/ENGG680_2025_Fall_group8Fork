import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast
import requests
import joblib
import os

class helper:
    def __init__(self):
        self.symptoms_df = None
        self.all_symptoms = None
        self.med_df = None
        self.clinic_df = None

    @staticmethod
    def clean_symptom(value):
        """Normalize symptom/disease strings for consistent matching."""
        if not isinstance(value, str):
            value = str(value)
        value = value.strip().lower()
        value = value.replace("-", " ")
        value = value.replace("_", " ")
        value = " ".join(value.split())
        value = value.replace(" ", "_")
        return value

    # ----------------------- Disease Model -----------------------
    def train_disease_model(self, df, model_type='random_forest'):
        """Train a multi-hot model for disease prediction.

        Parameters
        ----------
        df : DataFrame
            The symptoms dataframe. Supports two formats:
            1. Old format: columns are Disease, Symptom_1, Symptom_2, etc.
            2. New format: first column is disease name, other columns are symptoms with 1/0 values
        model_type : str
            Either 'random_forest' or 'naive_bayes'

        Returns
        -------
        tuple
            (model, disease_encoder, train_accuracy, test_accuracy, X_train, y_train, X_test, y_test)
        """
        self.symptoms_df = df.copy()
        
        # Detect dataset format
        if 'Disease' in self.symptoms_df.columns and any('Symptom' in col for col in self.symptoms_df.columns):
            # Old format: Disease column + Symptom_1, Symptom_2, etc.
            self.symptoms_df.fillna("none", inplace=True)
            
            # Clean all symptoms
            symptom_cols = [col for col in df.columns if "Symptom" in col]
            for col in symptom_cols:
                self.symptoms_df[col] = self.symptoms_df[col].astype(str).apply(self.clean_symptom)

            # Create list of all unique symptoms
            all_symptoms_series = pd.Series(dtype=str)
            for col in symptom_cols:
                all_symptoms_series = pd.concat([all_symptoms_series, self.symptoms_df[col].astype(str)])
            self.all_symptoms = sorted(all_symptoms_series.unique())
            if "none" in self.all_symptoms:
                self.all_symptoms.remove("none")

            # Multi-hot encode symptoms
            X = pd.DataFrame(0, index=self.symptoms_df.index, columns=self.all_symptoms)
            for col in symptom_cols:
                for i, val in enumerate(self.symptoms_df[col]):
                    if val != "none":
                        X.at[i, val] = 1
            
            disease_col = "Disease"
        else:
            # New format: first column is disease, rest are binary symptom columns
            disease_col = self.symptoms_df.columns[0]
            
            # All columns except first are symptoms
            symptom_cols = [col for col in self.symptoms_df.columns[1:]]
            
            # Clean symptom column names
            self.all_symptoms = [self.clean_symptom(col) for col in symptom_cols]
            
            # X is already in the right format (just rename columns and convert to int)
            X = self.symptoms_df[symptom_cols].astype(int)
            X.columns = self.all_symptoms

        # Encode diseases
        disease_encoder = LabelEncoder()
        y = disease_encoder.fit_transform(self.symptoms_df[disease_col])

        # Filter out classes with too few samples for stratified split (need at least 2)
        y_series = pd.Series(y)
        value_counts = y_series.value_counts()
        valid_classes = value_counts[value_counts >= 2].index.tolist()
        
        if len(valid_classes) < len(y_series.unique()):
            # Remove samples from classes with only 1 instance
            valid_mask = y_series.isin(valid_classes).values
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask]
            self.symptoms_df = self.symptoms_df[valid_mask].reset_index(drop=True)
            # Re-encode to compress label space
            disease_encoder = LabelEncoder()
            y = disease_encoder.fit_transform(self.symptoms_df[disease_col])

        # Stratified train-test split to ensure all classes appear in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the selected model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=25, random_state=42)
        elif model_type == 'naive_bayes':
            model = MultinomialNB()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest' or 'naive_bayes'")

        model.fit(X_train, y_train)

        # Calculate training and testing accuracy
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        return model, disease_encoder, train_accuracy, test_accuracy, X_train, y_train, X_test, y_test

    def save_model(self, model, disease_encoder, filepath='trained_model.joblib'):
        """Save trained model and encoder to disk using joblib for better compression."""
        model_data = {
            'model': model,
            'encoder': disease_encoder,
            'all_symptoms': self.all_symptoms,
            'symptoms_df': self.symptoms_df
        }
        joblib.dump(model_data, filepath, compress=3)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath='trained_model.joblib'):
        """Load trained model and encoder from disk.
        
        Returns
        -------
        tuple
            (model, disease_encoder)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' not found. Please train the model first.")
        
        model_data = joblib.load(filepath)
        
        self.all_symptoms = model_data['all_symptoms']
        self.symptoms_df = model_data['symptoms_df']
        
        print(f"✓ Model loaded from {filepath}")
        return model_data['model'], model_data['encoder']

    def predict_disease_probabilities(self, user_symptoms, model, disease_encoder, top_n=3, threshold=0.45):
        """Predict top probable diseases and return probabilities and medications."""
        # Clean user symptoms
        cleaned_user_symptoms = [self.clean_symptom(s) for s in user_symptoms.values()]

        # Multi-hot encode input
        input_data = {symptom: 0 for symptom in self.all_symptoms}
        for s in cleaned_user_symptoms:
            if s in input_data:
                input_data[s] = 1
        input_df = pd.DataFrame([input_data])

        # Predict probabilities
        proba = model.predict_proba(input_df)[0]
        classes = disease_encoder.inverse_transform(range(len(proba)))

        # First, collect all predictions with their probabilities
        all_predictions = list(zip(classes, proba))
        # Sort by probability descending and take top_n
        top_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:top_n]
        
        results = []
        for cls, p in top_predictions:
            # Apply threshold check
            if p < threshold:
                continue
                
            # Find common symptoms of this disease based on dataset format
            disease_col = self.symptoms_df.columns[0]
            
            if 'Disease' in self.symptoms_df.columns and any('Symptom' in col for col in self.symptoms_df.columns):
                # Old format
                disease_rows = self.symptoms_df[self.symptoms_df["Disease"] == cls]
                symptom_cols = [col for col in disease_rows.columns if "Symptom" in col]
                disease_symptoms = set()
                for col in symptom_cols:
                    disease_symptoms.update(
                        disease_rows[col].dropna().astype(str).apply(self.clean_symptom).tolist()
                    )
            else:
                # New format: get symptoms where value is 1
                disease_rows = self.symptoms_df[self.symptoms_df[disease_col] == cls]
                if not disease_rows.empty:
                    # Get first row for this disease
                    disease_row = disease_rows.iloc[0]
                    disease_symptoms = set()
                    for symptom in self.all_symptoms:
                        # Find original column name (before cleaning)
                        orig_cols = [col for col in self.symptoms_df.columns[1:] 
                                    if self.clean_symptom(col) == symptom]
                        if orig_cols and disease_row.get(orig_cols[0], 0) == 1:
                            disease_symptoms.add(symptom)
                else:
                    disease_symptoms = set()

            matched = [s for s in cleaned_user_symptoms if s in disease_symptoms]
            unmatched = [s for s in cleaned_user_symptoms if s not in disease_symptoms]

            # Get medications
            medications = self.get_medications(cls)

            results.append({
                "disease": cls,
                "probability": round(p, 3),
                "common_symptoms": sorted(disease_symptoms),
                "matched_symptoms": matched,
                "unmatched_symptoms": unmatched,
                "medications": medications
            })

        return results

    # ----------------------- Medications -----------------------
    def load_medications(self, med_csv_path):
        """Load medications CSV."""
        self.med_df = pd.read_csv(med_csv_path)
        self.med_df["Disease"] = self.med_df["Disease"].astype(str).apply(self.clean_symptom)

    def get_medications(self, disease_name):
        """Return list of medications for a disease."""
        if self.med_df is None:
            raise ValueError("Medication data not loaded. Run load_medications first.")
        disease_name_clean = self.clean_symptom(disease_name)
        row = self.med_df[self.med_df["Disease"] == disease_name_clean]
        if row.empty:
            return []

        meds_str = row["Medication"].values[0]
        try:
            meds_list = ast.literal_eval(meds_str)
        except Exception:
            meds_list = [meds_str]
        return meds_list

    # ----------------------- Clinic Inventory -----------------------
    def load_clinic_inventory(self, clinic_csv_path):
        """Load clinic inventory CSV."""
        self.clinic_df = pd.read_csv(clinic_csv_path)
        self.clinic_df.columns = [self.clean_symptom(c) for c in self.clinic_df.columns]

    def find_clinics_for_predicted_diseases(self, prediction_results):
        """Return clinics that have suggested medications for predicted diseases."""
        if self.clinic_df is None:
            raise ValueError("Clinic inventory not loaded. Run load_clinic_inventory first.")

        disease_clinics = {}

        for result in prediction_results:
            disease_name = result['disease']
            meds = result.get('medications', [])
            # Clean medications to match clinic columns
            meds_cleaned = [self.clean_symptom(m) for m in meds]
            meds_in_columns = [m for m in meds_cleaned if m in self.clinic_df.columns]

            clinics_list = []
            for _, row in self.clinic_df.iterrows():
                meds_available = [m for m in meds_in_columns if row[m] > 0]
                if meds_available:
                    clinics_list.append({
                        "clinic_name": row["clinic_name"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "available_medicines": meds_available,
                        "num_available": len(meds_available)
                    })

            # Sort by most medicines available
            clinics_list = sorted(clinics_list, key=lambda x: x["num_available"], reverse=True)
            disease_clinics[disease_name] = clinics_list

        return disease_clinics

    def get_top_clinics_by_road_distance(self, user_lat, user_lon, clinics):
        """
        Uses OSRM public API to calculate real road-driving distances
        between the user location and clinics that have required medications.

        Parameters
        ----------
        user_lat : float
            User latitude
        user_lon : float
            User longitude
        clinics : dict
            Dictionary of clinic_name -> list of clinic entries containing coordinates and meds

        Returns
        -------
        list
            Top 3 nearest clinics sorted by road distance
        """

        clinic_distances = []

        # Flatten clinics into a single list
        flat_clinics = []
        for valuelist in clinics.values():
            for clinic in valuelist:
                flat_clinics.append(clinic)

        for clinic in flat_clinics:
            c_lat = clinic["latitude"]
            c_lon = clinic["longitude"]
            c_name = clinic["clinic_name"]

            # OSRM API URL (lon,lat format)
            url = f"http://router.project-osrm.org/route/v1/driving/{user_lon},{user_lat};{c_lon},{c_lat}?overview=false"

            try:
                response = requests.get(url, timeout=5)
                data = response.json()

                if "routes" in data and len(data["routes"]) > 0:
                    distance_meters = data["routes"][0]["distance"]
                    distance_km = distance_meters / 1000.0
                else:
                    distance_km = float("inf")

            except Exception:
                # If API fails, fallback to infinite distance
                distance_km = float("inf")

            clinic_distances.append({
                "clinic_name": c_name,
                "distance_km": distance_km,
                "latitude": c_lat,
                "longitude": c_lon,
                "medications_available": clinic.get("available_medicines", [])
            })

        # Sort by distance
        clinic_distances.sort(key=lambda x: x["distance_km"])

        # Return top 3 clinics
        return clinic_distances[:3]
