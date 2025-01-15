import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from typing import Dict, Any, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malaria_prediction_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MalariaPredictionApp:
    def __init__(self, model_path: str = 'malaria_model.pkl'):
        """Initialize the prediction app with the trained model."""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            self.scaler = StandardScaler()
            self.le = LabelEncoder()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def get_float_input(self, prompt: str, min_val: float, max_val: float) -> float:
        """Get validated float input from user."""
        while True:
            try:
                value = float(input(prompt))
                if min_val <= value <= max_val:
                    return value
                print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")

    def get_categorical_input(self, prompt: str, options: list) -> str:
        """Get validated categorical input from user."""
        while True:
            print(prompt)
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            try:
                choice = int(input("Enter your choice (number): "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                print("Please select a valid option")
            except ValueError:
                print("Please enter a valid number")

    def get_binary_input(self, prompt: str) -> int:
        """Get validated yes/no input from user."""
        while True:
            response = input(f"{prompt} (yes/no): ").lower()
            if response in ['yes', 'y']:
                return 1
            elif response in ['no', 'n']:
                return 0
            print("Please enter 'yes' or 'no'")

    def get_user_inputs(self) -> Dict[str, Any]:
        """Collect all required inputs from user."""
        print("\n=== Malaria Outbreak Prediction System ===")
        print("Please provide the following information:\n")

        inputs = {}
        
        # Environmental factors
        print("\n--- Environmental Factors ---")
        inputs['temperature'] = self.get_float_input(
            "Enter average temperature (°C, 15-40): ", 15, 40)
        inputs['humidity'] = self.get_float_input(
            "Enter relative humidity (%%, 0-100): ", 0, 100)
        inputs['rainfall'] = self.get_float_input(
            "Enter monthly rainfall (mm, 0-1000): ", 0, 1000)
        inputs['wind_speed'] = self.get_float_input(
            "Enter average wind speed (km/h, 0-50): ", 0, 50)
        inputs['weather_pattern'] = self.get_categorical_input(
            "\nSelect current weather pattern:", ['Rainy', 'Dry', 'Mixed'])
        
        # Healthcare and demographic factors
        print("\n--- Healthcare & Demographic Factors ---")
        inputs['vaccination_rate'] = self.get_float_input(
            "Enter local vaccination rate (%%, 0-100): ", 0, 100)
        inputs['housing_type'] = self.get_categorical_input(
            "\nSelect predominant housing type:", ['Urban', 'Rural', 'Suburban'])
        inputs['socio_economic_status'] = self.get_categorical_input(
            "\nSelect socio-economic status:", ['Low', 'Medium', 'High'])
        inputs['bed_capacity'] = self.get_float_input(
            "Enter hospital bed capacity per 10,000 people (0-100): ", 0, 100)
        inputs['medical_staff'] = self.get_float_input(
            "Enter medical staff per 10,000 people (0-50): ", 0, 50)
        inputs['population_density'] = self.get_float_input(
            "Enter population density (people/km², 0-5000): ", 0, 5000)
        inputs['employment_rate'] = self.get_float_input(
            "Enter employment rate (%%, 0-100): ", 0, 100)
        inputs['land_use_type'] = self.get_categorical_input(
            "\nSelect predominant land use type:", ['Agricultural', 'Residential', 'Industrial'])
        
        # Binary factors
        print("\n--- Additional Factors ---")
        inputs['is_rainy_season'] = self.get_binary_input("Is it currently rainy season?")
        inputs['access_to_water'] = self.get_binary_input("Is there reliable access to clean water?")
        inputs['access_to_healthcare'] = self.get_binary_input("Is there easy access to healthcare facilities?")

        return inputs

    def preprocess_inputs(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Preprocess the user inputs to match model requirements."""
        # Create DataFrame with original features in correct order
        df = pd.DataFrame([{
            'temperature': inputs['temperature'],
            'humidity': inputs['humidity'],
            'rainfall': inputs['rainfall'],
            'wind_speed': inputs['wind_speed'],
            'vaccination_rate': inputs['vaccination_rate'],
            'bed_capacity': inputs['bed_capacity'],
            'medical_staff': inputs['medical_staff'],
            'population_density': inputs['population_density'],
            'employment_rate': inputs['employment_rate'],
            'socio_economic_status': inputs['socio_economic_status'],
            'housing_type': inputs['housing_type'],
            'weather_pattern': inputs['weather_pattern'],
            'land_use_type': inputs['land_use_type'],
            'is_rainy_season': inputs['is_rainy_season'],
            'access_to_water': inputs['access_to_water'],
            'access_to_healthcare': inputs['access_to_healthcare']
        }])

        # Convert categorical variables using LabelEncoder
        categorical_columns = ['socio_economic_status', 'housing_type', 'weather_pattern', 'land_use_type']
        for col in categorical_columns:
            self.le.fit(df[col])
            df[col] = self.le.transform(df[col])

        # Scale numerical features
        numerical_columns = ['temperature', 'humidity', 'rainfall', 'wind_speed', 'vaccination_rate',
                           'bed_capacity', 'medical_staff', 'population_density', 'employment_rate']
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])

        # Ensure column order matches training data
        feature_order = [
            'temperature', 'humidity', 'rainfall', 'wind_speed', 'vaccination_rate',
            'bed_capacity', 'medical_staff', 'population_density', 'employment_rate',
            'socio_economic_status', 'housing_type', 'weather_pattern', 'land_use_type',
            'is_rainy_season', 'access_to_water', 'access_to_healthcare'
        ]
        df = df[feature_order]
        
        return df

    def make_prediction(self, inputs: Dict[str, Any]) -> Tuple[int, float]:
        """Make prediction using the trained model."""
        try:
            # Preprocess the inputs
            processed_inputs = self.preprocess_inputs(inputs)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(processed_inputs)[0]
            prediction = self.model.predict(processed_inputs)[0]
            
            return prediction, prediction_proba[1]
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def run(self):
        """Run the prediction application."""
        try:
            while True:
                # Get user inputs
                inputs = self.get_user_inputs()
                
                # Make prediction
                prediction, probability = self.make_prediction(inputs)
                
                # Display results
                print("\n=== Prediction Results ===")
                if prediction == 1:
                    print(f"WARNING: High risk of malaria outbreak")
                    print(f"Confidence: {probability:.1%}")
                else:
                    print(f"Low risk of malaria outbreak")
                    print(f"Confidence: {1-probability:.1%}")
                
                # Ask if user wants to make another prediction
                if not self.get_binary_input("\nWould you like to make another prediction?"):
                    print("\nThank you for using the Malaria Outbreak Prediction System!")
                    break
                
        except KeyboardInterrupt:
            print("\n\nExiting the application...")
        except Exception as e:
            logger.error(f"Application error: {e}")
            print("\nAn error occurred. Please check the logs for details.")

if __name__ == "__main__":
    try:
        app = MalariaPredictionApp()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print("Failed to start the application. Please check the logs for details.")