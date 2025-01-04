import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib
from datetime import datetime

class MalariaOutbreakPredictor:
    def __init__(self, host, user, password, database):
        """Initialize the predictor with database credentials"""
        self.db_config = {
            'host': "127.0.0.1",
            'user': "admin",
            'password': "admin",
            'database': "malaria"
        }
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def create_engine(self):
        """Create SQLAlchemy engine"""
        return create_engine(
            f"mysql+mysqlconnector://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}/{self.db_config['database']}"
        )

    def fetch_data(self):
        """Fetch data using SQLAlchemy"""
        try:
            engine = self.create_engine()
            query = """
            SELECT 
                f.Outbreak_status,
                w.Temperature,
                w.Humidity,
                w.Rainfall,
                w.Wind_speed,
                w.Weather_pattern,
                l.Population_density,
                l.Distance_to_water,
                l.Terrain_type,
                d.Is_rainy_season,
                d.Season,
                e.Population_level,
                e.Vegetation_type,
                h.Antimalarial_stock,
                h.Distance_to_communities,
                h.Medical_staff,
                p.Spray_coverage,
                p.Nets_distributed,
                dem.Vaccination_rate,
                dem.Housing_type,
                i.Road_quality,
                i.Access_to_water,
                i.Access_to_electricity
            FROM FACT_MALARIA_CASES f
            JOIN DIM_WEATHER w ON f.Weather_key = w.Weather_key
            JOIN DIM_LOCATION l ON f.Location_key = l.Location_key
            JOIN DIM_DATES d ON f.Date_key = d.Date_key
            JOIN DIM_ENVIRONMENT e ON f.Environment_key = e.Environment_key
            JOIN DIM_HEALTHCARE h ON f.Healthcare_key = h.Healthcare_key
            JOIN DIM_PREVENTION p ON f.Prevention_key = p.Prevention_key
            JOIN DIM_DEMOGRAPHICS dem ON f.Demographic_key = dem.Demographic_key
            JOIN DIM_INFRASTRUCTURE i ON f.Infrastructure_key = i.Infrastructure_key
            """
            
            df = pd.read_sql(query, engine)
            
            # Convert Outbreak_status to binary
            df['Outbreak_status'] = df['Outbreak_status'].astype(int)
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocess the data with feature engineering"""
        # Handle categorical variables
        categorical_features = [
            'Weather_pattern', 'Terrain_type', 'Season',
            'Population_level', 'Vegetation_type', 'Housing_type',
            'Road_quality'
        ]
        
        # Encode categorical variables
        for feature in categorical_features:
            self.encoders[feature] = LabelEncoder()
            df[feature] = self.encoders[feature].fit_transform(df[feature])
        
        # Convert boolean features to integers
        boolean_features = ['Is_rainy_season', 'Access_to_water', 'Access_to_electricity']
        for feature in boolean_features:
            df[feature] = df[feature].astype(int)
        
        # Create derived features
        df['Rain_Temperature_Index'] = df['Rainfall'] * df['Temperature']
        df['Healthcare_Access_Score'] = (df['Medical_staff'] / 
                                       np.maximum(df['Population_density'], 1))  # Avoid division by zero
        df['Prevention_Coverage'] = (df['Spray_coverage'] + 
                                   df['Nets_distributed']/np.maximum(df['Population_density'], 1))/2
        df['Environmental_Risk'] = df['Distance_to_water'] * df['Humidity']
        df['Infrastructure_Score'] = (df['Access_to_water'].astype(int) + 
                                    df['Access_to_electricity'].astype(int))/2
        
        # Scale numerical features
        numerical_features = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                            if col != 'Outbreak_status']
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        return df

    def train_model(self, X, y):
        """Train and optimize the Random Forest model"""
        # Convert target to int type
        y = y.astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Simplified parameter grid for initial testing
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42)
        
        # Perform grid search with error debugging enabled
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1, error_score='raise'
        )
        
        try:
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise
        
        return X_train, X_test, y_train, y_test

    # ... [rest of the class methods remain the same] ...

def main():
    # Initialize predictor
    predictor = MalariaOutbreakPredictor(
        host="localhost",
        user="your_username",
        password="your_password",
        database="malaria"
    )
    
    # Fetch and prepare data
    print("Fetching data...")
    df = predictor.fetch_data()
    if df is None:
        return
    
    print("Data shape:", df.shape)
    print("\nTarget variable distribution:")
    print(df['Outbreak_status'].value_counts(normalize=True))
    
    # Preprocess data
    print("\nPreprocessing data...")
    processed_df = predictor.preprocess_data(df)
    
    # Separate features and target
    X = processed_df.drop('Outbreak_status', axis=1)
    y = processed_df['Outbreak_status']
    
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)
    
    # Train model
    print("\nTraining model...")
    try:
        X_train, X_test, y_train, y_test = predictor.train_model(X, y)
        
        # Evaluate model
        print("\nEvaluating model...")
        feature_importance = predictor.evaluate_model(X_test, y_test)
        
        # Save model
        predictor.save_model()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()