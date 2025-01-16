import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import time
from typing import Dict, Any

class OutbreakPredictor:
    def __init__(self, model_path='malaria_model.pkl'):
        # Load the trained model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)
        
        self.scaler = StandardScaler()
        self.label_encoders = {
            'socio_economic_status': ['Low', 'Medium', 'High'],
            'housing_type': ['Urban', 'Rural', 'Suburban'],
            'weather_pattern': ['Rainy', 'Dry', 'Mixed'],
            'land_use_type': ['Agricultural', 'Residential', 'Industrial']
        }

    def get_user_input(self) -> Dict[str, Any]:
        """Get input from user with validation"""
        data = {}
        
        # Numeric inputs
        numeric_fields = {
            'temperature': ('Enter temperature (°C)', 0, 50),
            'humidity': ('Enter humidity (%)', 0, 100),
            'rainfall': ('Enter rainfall (mm)', 0, 1000),
            'wind_speed': ('Enter wind speed (km/h)', 0, 200),
            'vaccination_rate': ('Enter vaccination rate (%)', 0, 100),
            'bed_capacity': ('Enter hospital bed capacity', 0, 10000),
            'medical_staff': ('Enter number of medical staff', 0, 1000),
            'population_density': ('Enter population density (per km²)', 0, 50000),
            'employment_rate': ('Enter employment rate (%)', 0, 100)
        }

        for field, (prompt, min_val, max_val) in numeric_fields.items():
            while True:
                try:
                    value = float(input(f"{prompt}: "))
                    if min_val <= value <= max_val:
                        data[field] = value
                        break
                    else:
                        print(f"Please enter a value between {min_val} and {max_val}")
                except ValueError:
                    print("Please enter a valid number")

        # Categorical inputs
        for field, options in self.label_encoders.items():
            while True:
                print(f"\nAvailable options for {field}: {', '.join(options)}")
                value = input(f"Enter {field}: ").capitalize()
                if value in options:
                    data[field] = value
                    break
                else:
                    print(f"Please enter one of: {', '.join(options)}")

        # Binary inputs
        binary_fields = {
            'is_rainy_season': 'Is it currently rainy season? (yes/no): ',
            'access_to_water': 'Is there access to clean water? (yes/no): ',
            'access_to_healthcare': 'Is there access to healthcare? (yes/no): '
        }

        for field, prompt in binary_fields.items():
            while True:
                value = input(prompt).lower()
                if value in ['yes', 'no']:
                    data[field] = 1 if value == 'yes' else 0
                    break
                else:
                    print("Please enter 'yes' or 'no'")

        return data

    def engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer additional features from input data"""
        engineered = data.copy()
        
        # Calculate derived features
        engineered['temp_humidity_interaction'] = data['temperature'] * data['humidity']
        engineered['healthcare_access_score'] = (data['bed_capacity'] / 100 +
                                               data['medical_staff'] / 20 +
                                               data['access_to_healthcare'] * 2)
        engineered['environmental_risk'] = ((data['temperature'] > 28) +
                                          (data['humidity'] > 75) +
                                          (data['rainfall'] > 150))
        engineered['high_risk_conditions'] = int(
            (engineered['environmental_risk'] >= 2) and
            (data['access_to_healthcare'] == 0)
        )
        
        return engineered

    def preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess the input data for prediction"""
        # Engineer features
        data = self.engineer_features(data)
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, categories in self.label_encoders.items():
            le = LabelEncoder()
            le.fit(categories)
            df[col] = le.transform(df[col])
        
        # Scale numerical features
        numerical_cols = [col for col in df.columns if col not in self.label_encoders]
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df

    def predict(self, data: Dict[str, Any]) -> tuple:
        """Make prediction and return probability"""
        df = self.preprocess_data(data)
        probability = self.model.predict_proba(df)[0][1]
        prediction = self.model.predict(df)[0]
        return prediction, probability

def display_prediction_result(prediction: int, probability: float):
    """Display the prediction result with a simple animation"""
    print("\nAnalyzing data", end="")
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")
    
    time.sleep(1)
    print("=" * 50)
    if prediction == 1:
        print("\n🚨 WARNING: High Risk of Malaria Outbreak 🚨")
        print(f"Probability: {probability:.1%}")
        print("\nRecommended Actions:")
        print("1. Alert local health authorities")
        print("2. Increase surveillance")
        print("3. Prepare medical supplies")
        print("4. Consider preventive measures")
    else:
        print("\n✅ Low Risk of Malaria Outbreak")
        print(f"Probability: {probability:.1%}")
        print("\nRecommended Actions:")
        print("1. Continue regular monitoring")
        print("2. Maintain preventive measures")
    print("=" * 50)

def main():
    print("\n=== Malaria Outbreak Prediction System ===")
    print("This system predicts the likelihood of malaria outbreaks")
    print("based on environmental and social factors.\n")

    predictor = OutbreakPredictor()
    
    while True:
        print("\nPlease enter the following information:")
        data = predictor.get_user_input()
        
        prediction, probability = predictor.predict(data)
        display_prediction_result(prediction, probability)
        
        while True:
            again = input("\nWould you like to make another prediction? (yes/no): ").lower()
            if again in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")
            
        if again == 'no':
            print("\nThank you for using the Malaria Outbreak Prediction System!")
            break
        print("\n" + "=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        sys.exit(0)