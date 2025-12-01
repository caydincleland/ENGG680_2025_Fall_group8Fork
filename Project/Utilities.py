import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast
import requests

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
            The symptoms dataframe
        model_type : str
            Either 'random_forest' or 'naive_bayes'
        
        Returns
        -------
        tuple
            (model, disease_encoder, accuracy, X_test, y_test)
        """
        self.symptoms_df = df.copy()
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

        # Encode diseases
        disease_encoder = LabelEncoder()
        y = disease_encoder.fit_transform(self.symptoms_df["Disease"])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the selected model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=300, random_state=42)
        elif model_type == 'naive_bayes':
            model = MultinomialNB()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest' or 'naive_bayes'")
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return model, disease_encoder, accuracy, X_test, y_test

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

        results = []
        for cls, p in zip(classes, proba):
            if p >= threshold:
                # Find common symptoms of this disease
                disease_rows = self.symptoms_df[self.symptoms_df["Disease"] == cls]
                symptom_cols = [col for col in disease_rows.columns if "Symptom" in col]
                disease_symptoms = set()
                for col in symptom_cols:
                    disease_symptoms.update(
                        disease_rows[col].dropna().astype(str).apply(self.clean_symptom).tolist()
                    )

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

        # Sort by probability descending and return top_n
        results = sorted(results, key=lambda x: x['probability'], reverse=True)[:top_n]
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
