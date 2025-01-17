import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import time
from typing import Dict, Any, Tuple

class OutbreakPredictor:
    def __init__(self, model_path='malaria_model.pkl'):
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
        
        # Define probability thresholds for risk levels
        self.risk_thresholds = {
            'low': 0.30,
            'moderate': 0.55,
            'high': 0.75,
            'very_high': 0.9
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

    def engineer_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer additional features from input data"""
        engineered = data.copy()
        
        # Calculate derived features with adjusted weights
        engineered['temp_humidity_interaction'] = (data['temperature'] * data['humidity']) / (50 * 100)  # Normalize to 0-1
        engineered['healthcare_access_score'] = (
            (data['bed_capacity'] / 1000) * 0.3 +  # Normalize to reasonable max capacity
            (data['medical_staff'] / 100) * 0.3 +  # Normalize to reasonable staff size
            (data['access_to_healthcare']) * 0.4   # Binary factor weighted more heavily
        )
        
        # Environmental risk calculation (0-1 scale)
        engineered['environmental_risk'] = (
            (int(data['temperature'] > 28) * 0.4) +  # Temperature threshold
            (int(data['humidity'] > 65) * 0.3) +     # Humidity threshold
            (int(data['rainfall'] > 150) * 0.3)      # Rainfall threshold
        )
        
        # High risk conditions (0-1 scale)
        engineered['high_risk_conditions'] = float(
            (engineered['environmental_risk'] >= 0.6) and  # High environmental risk
            (data['access_to_healthcare'] == 0)           # No healthcare access
        )
        
        return engineered

    def calculate_final_probability(self, components: Dict[str, float]) -> float:
        """Calculate final probability based on weighted risk components"""
        # Define weights for each component
        weights = {
            'Environmental Risk': 0.35,
            'Healthcare Access': -0.25,  # Negative because higher access reduces risk
            'Climate Conditions': 0.40
        }
        
        # Calculate weighted sum
        weighted_sum = 0
        for component, value in components.items():
            weight = weights.get(component, 0)
            if component == 'Healthcare Access':
                weighted_sum += weight * (1 - value)  # Invert healthcare access
            else:
                weighted_sum += weight * value
        
        # Convert to probability (0-1 scale)
        probability = np.clip(weighted_sum + 0.5, 0, 1)  # Add 0.5 to center around middle
        return probability

    def predict(self, data: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction and return risk level and probability"""
        df = self.preprocess_data(data)
        
        # Calculate risk components
        engineered = self.engineer_features(data)
        risk_components = {
            'Environmental Risk': engineered['environmental_risk'],
            'Healthcare Access': engineered['healthcare_access_score'],
            'Climate Conditions': engineered['temp_humidity_interaction']
        }
        
        # Calculate final probability based on components
        probability = self.calculate_final_probability(risk_components)
        
        # Determine risk level based on probability
        if probability >= self.risk_thresholds['very_high']:
            risk_level = 'very_high'
        elif probability >= self.risk_thresholds['high']:
            risk_level = 'high'
        elif probability >= self.risk_thresholds['moderate']:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return risk_level, probability, risk_components

def display_prediction_result(risk_level: str, probability: float, risk_components: Dict[str, float]):
    """Enhanced display of the prediction result"""
    print("\nAnalyzing data", end="")
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")
    
    time.sleep(1)
    print("=" * 60)
    
    # Risk level indicators and colors
    risk_indicators = {
        'very_high': ('🔴', 'VERY HIGH RISK'),
        'high': ('🟠', 'HIGH RISK'),
        'moderate': ('🟡', 'MODERATE RISK'),
        'low': ('🟢', 'LOW RISK')
    }
    
    icon, level_text = risk_indicators[risk_level]
    print(f"\n{icon} Malaria Outbreak Risk Level: {level_text}")
    print(f"Risk Probability: {probability:.1%}\n")
    
    # Display risk components
    print("Risk Component Analysis:")
    for component, value in risk_components.items():
        bar_length = 20
        filled = int(value * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"{component:20} [{bar}] {value:.1%}")
    
    print("\nRecommended Actions:")
    if risk_level == 'very_high':
        print("🚨 IMMEDIATE ACTION REQUIRED:")
        print("1. Alert health authorities immediately")
        print("2. Mobilize emergency medical resources")
        print("3. Implement intensive surveillance")
        print("4. Activate emergency response protocols")
    elif risk_level == 'high':
        print("⚠️ URGENT ACTION REQUIRED:")
        print("1. Notify health authorities")
        print("2. Increase surveillance measures")
        print("3. Prepare medical supplies")
        print("4. Review emergency protocols")
    elif risk_level == 'moderate':
        print("⚠️ HEIGHTENED VIGILANCE REQUIRED:")
        print("1. Enhance monitoring systems")
        print("2. Review medical preparedness")
        print("3. Update prevention measures")
        print("4. Alert relevant stakeholders")
    else:  # low
        print("✓ STANDARD MEASURES:")
        print("1. Maintain routine surveillance")
        print("2. Continue preventive measures")
        print("3. Regular monitoring")
    
    print("=" * 60)

def main():
    print("\n=== Enhanced Malaria Outbreak Prediction System ===")
    print("This system predicts the likelihood of malaria outbreaks")
    print("based on environmental and social factors.\n")

    predictor = OutbreakPredictor()
    
    while True:
        print("\nPlease enter the following information:")
        data = predictor.get_user_input()
        
        risk_level, probability, risk_components = predictor.predict(data)
        display_prediction_result(risk_level, probability, risk_components)
        
        while True:
            again = input("\nWould you like to make another prediction? (yes/no): ").lower()
            if again in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")
            
        if again == 'no':
            print("\nThank you for using the Malaria Outbreak Prediction System!")
            break
        print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        sys.exit(0)