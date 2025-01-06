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
            'host': host,
            'user': user,
            'password': password,
            'database': database
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
                                       np.maximum(df['Population_density'], 1))
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

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model using various metrics and visualizations"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
        
        # Create confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Save feature importance to CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
        
        print("\nFeature Importance Analysis:")
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance.to_dict()

    def save_model(self, filename=None):
        """Save the trained model and preprocessing objects"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
            
        if filename is None:
            filename = f"malaria_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Save the model
        joblib.dump(self.model, f'{filename}_model.joblib')
        
        # Save the encoders
        joblib.dump(self.encoders, f'{filename}_encoders.joblib')
        
        # Save the scaler
        joblib.dump(self.scaler, f'{filename}_scaler.joblib')
        
        print(f"\nModel and preprocessing objects saved with prefix: {filename}")

def main():
    """Main execution function"""
    try:
        # Initialize predictor
        predictor = MalariaOutbreakPredictor(
            host="127.0.0.1",  # Update these with your actual database credentials
            user="admin",
            password="admin",
            database="malaria"
        )
        
        # Fetch data
        print("\nFetching data...")
        df = predictor.fetch_data()
        if df is None:
            print("Error: Could not fetch data from database")
            return
        
        print(f"\nData shape: {df.shape}")
        print("\nTarget variable distribution:")
        print(df['Outbreak_status'].value_counts(normalize=True))
        
        # Preprocess data
        print("\nPreprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        # Separate features and target
        X = processed_df.drop('Outbreak_status', axis=1)
        y = processed_df['Outbreak_status']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Train model
        print("\nTraining model...")
        X_train, X_test, y_train, y_test = predictor.train_model(X, y)
        
        # Evaluate model
        print("\nEvaluating model...")
        feature_importance = predictor.evaluate_model(X_test, y_test)
        
        # Save model
        print("\nSaving model...")
        predictor.save_model()
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    main()