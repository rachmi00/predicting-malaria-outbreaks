import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

class MalariaOutbreakPredictor:
    def __init__(self):
        # Risk thresholds
        self.HIGH_RISK_THRESHOLD = 0.7
        self.MODERATE_RISK_THRESHOLD = 0.4
        
        # Load the model and scalers (placeholder for actual model)
        # In practice, you would load your trained model here
        # self.model = pickle.load(open('malaria_model.pkl', 'rb'))
        self.scaler = StandardScaler()
        
        # Define feature requirements with units
        self.features = {
            'temperature': {'prompt': 'Average temperature (°C)', 'type': float, 'range': (15, 40)},
            'rainfall': {'prompt': 'Monthly rainfall (mm)', 'type': float, 'range': (0, 1000)},
            'humidity': {'prompt': 'Relative humidity (%)', 'type': float, 'range': (0, 100)},
            'population_density': {'prompt': 'Population density (people/km²)', 'type': float, 'range': (0, 1000)},
            'healthcare_facilities': {'prompt': 'Number of healthcare facilities in the area', 'type': int, 'range': (0, 100)},
            'mosquito_breeding_sites': {'prompt': 'Number of identified mosquito breeding sites', 'type': int, 'range': (0, 1000)},
            'bed_net_coverage': {'prompt': 'Bed net coverage (%)', 'type': float, 'range': (0, 100)},
            'distance_to_water': {'prompt': 'Distance to nearest water body (km)', 'type': float, 'range': (0, 100)},
            'elevation': {'prompt': 'Elevation above sea level (m)', 'type': float, 'range': (0, 3000)},
            'vegetation_density': {'prompt': 'Vegetation density index (0-1)', 'type': float, 'range': (0, 1)},
        }

    def get_user_input(self):
        """Collect user input for all features with validation"""
        data = {}
        print("\n=== Malaria Outbreak Risk Prediction System ===")
        print("Please enter the following information:")
        
        for feature, props in self.features.items():
            while True:
                try:
                    value = props['type'](input(f"\n{props['prompt']}: "))
                    min_val, max_val = props['range']
                    
                    if min_val <= value <= max_val:
                        data[feature] = value
                        break
                    else:
                        print(f"Value must be between {min_val} and {max_val}")
                except ValueError:
                    print(f"Please enter a valid {props['type'].__name__}")
        
        return data

    def preprocess_data(self, data):
        """Preprocess the input data"""
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(df)
        return scaled_data

    def predict_risk(self, data):
        """Predict outbreak risk and classify it"""
        # For demonstration, using a simplified risk calculation
        # In practice, you would use your trained model here
        scaled_data = self.preprocess_data(data)
        
        # Placeholder prediction logic (replace with actual model prediction)
        risk_score = np.mean(scaled_data) * 0.7 + np.random.random() * 0.3
        risk_score = max(0, min(1, risk_score))  # Ensure score is between 0 and 1
        
        # Classify risk level
        if risk_score >= self.HIGH_RISK_THRESHOLD:
            risk_level = "HIGH"
            color = "\033[91m"  # Red
        elif risk_score >= self.MODERATE_RISK_THRESHOLD:
            risk_level = "MODERATE"
            color = "\033[93m"  # Yellow
        else:
            risk_level = "LOW"
            color = "\033[92m"  # Green
            
        return risk_score, risk_level, color

    def display_results(self, data, risk_score, risk_level, color):
        """Display prediction results"""
        print("\n=== Prediction Results ===")
        print(f"Date of prediction: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nInput Parameters:")
        for feature, value in data.items():
            print(f"{feature.replace('_', ' ').title()}: {value}")
        
        print("\nRisk Assessment:")
        print(f"Risk Score: {risk_score:.2%}")
        print(f"Risk Level: {color}{risk_level}\033[0m")
        
        # Print recommendations based on risk level
        print("\nRecommendations:")
        if risk_level == "HIGH":
            print("- Immediate activation of emergency response protocols")
            print("- Increase surveillance and monitoring")
            print("- Deploy additional preventive measures")
            print("- Alert healthcare facilities")
        elif risk_level == "MODERATE":
            print("- Enhanced monitoring of the situation")
            print("- Review and reinforce preventive measures")
            print("- Prepare response resources")
        else:
            print("- Maintain routine surveillance")
            print("- Continue regular preventive measures")
            print("- Monitor for any changes in conditions")

def main():
    predictor = MalariaOutbreakPredictor()
    
    while True:
        # Get user input
        data = predictor.get_user_input()
        
        # Make prediction
        risk_score, risk_level, color = predictor.predict_risk(data)
        
        # Display results
        predictor.display_results(data, risk_score, risk_level, color)
        
        # Ask if user wants to make another prediction
        if input("\nWould you like to make another prediction? (y/n): ").lower() != 'y':
            break
    
    print("\nThank you for using the Malaria Outbreak Prediction System!")

if __name__ == "__main__":
    main()