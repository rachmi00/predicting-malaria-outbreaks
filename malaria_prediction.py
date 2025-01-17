import mysql.connector
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sqlalchemy import create_engine

class MalariaPredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def connect_to_db(self):
        return create_engine('mysql+mysqlconnector://admin:admin@127.0.0.1/malaria')

    def preprocess_dates(self, df):
        """Convert datetime columns to useful numerical features"""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        for col in date_columns:
            # Extract useful components from datetime
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            # Drop the original datetime column
            df = df.drop(columns=[col])
        
        return df

    def get_real_data(self):
        """Fetch and preprocess real data from database"""
        try:
            engine = self.connect_to_db()
            
            tables = [
                'dim_dates', 'dim_demographics', 'dim_environment', 
                'dim_health_initiatives', 'dim_healthcare', 'dim_infrastructure',
                'dim_location', 'dim_prevention', 'dim_socioeconomic',
                'dim_weather', 'fact_malaria_cases'
            ]
            
            dfs = {}
            for table in tables:
                query = f"SELECT * FROM {table}"
                dfs[table] = pd.read_sql(query, engine)
            
            engine.dispose()
            
            # Combine fact table with dimension tables
            fact_df = dfs.pop('fact_malaria_cases')
            merged_df = fact_df.copy()
            
            join_keys = {
                'dim_dates': 'Date_key',
                'dim_demographics': 'Demographic_key',
                'dim_environment': 'Environment_key',
                'dim_health_initiatives': 'Initiative_key',
                'dim_healthcare': 'Healthcare_key',
                'dim_infrastructure': 'Infrastructure_key',
                'dim_location': 'Location_key',
                'dim_prevention': 'Prevention_key',
                'dim_socioeconomic': 'Socioeconomic_key',
                'dim_weather': 'Weather_key',
            }
            
            for dim_table, key in join_keys.items():
                if dim_table in dfs:
                    merged_df = merged_df.merge(dfs[dim_table], on=key, how='left')
            
            # Clean and preprocess the merged dataframe
            merged_df = self.clean_dataframe(merged_df)
            merged_df = self.preprocess_dates(merged_df)
            
            return merged_df
            
        except Exception as e:
            print(f"Error fetching real data: {str(e)}")
            return None

    def clean_dataframe(self, df):
        """Clean dataframe by handling missing values and invalid data"""
        print("Cleaning dataframe...")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For numerical columns, fill NaN with median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill NaN with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Ensure outbreak_status is binary (0 or 1)
        if 'outbreak_status' in df.columns:
            df['outbreak_status'] = df['outbreak_status'].map({
                True: 1, False: 0, 
                'true': 1, 'false': 0,
                '1': 1, '0': 0,
                1: 1, 0: 0
            }).fillna(0).astype(int)
        
        return df

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data to augment real data"""
        # Generate dates without including them in the final dataset
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_samples)
        
        synthetic_data = {
            'temperature': np.random.normal(30, 3, n_samples),  # Increased mean temperature
        
        # Higher humidity levels favor mosquito breeding
            'humidity': np.random.uniform(75, 95, n_samples),  # Increased humidity range
        
        # More rainfall creates more breeding sites
            'rainfall': np.random.exponential(200, n_samples),  # Increased mean rainfall
        
        # Other environmental and healthcare factors
            'wind_speed': np.random.normal(8, 2, n_samples),
            'vaccination_rate': np.random.uniform(0, 60, n_samples),  # Lower vaccination rates
            'bed_capacity': np.random.randint(10, 500, n_samples),   # Limited healthcare capacity
            'medical_staff': np.random.randint(5, 100, n_samples),   # Limited medical staff
            'population_density': np.random.exponential(2000, n_samples),  # Higher population density
            'employment_rate': np.random.uniform(40, 70, n_samples),  # Lower employment rate
            'outbreak_status': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        # Add date-related features instead of actual dates
        synthetic_data['year'] = dates.year
        synthetic_data['month'] = dates.month
        synthetic_data['day'] = dates.day
        synthetic_data['dayofweek'] = dates.dayofweek
        
        df = pd.DataFrame(synthetic_data)
        return self.clean_dataframe(df)

    def preprocess_combined_data(self, real_df, synthetic_df):
        """Preprocess and combine real and synthetic data"""
        print("Preprocessing and combining datasets...")
        
        if real_df is None:
            print("Warning: No real data available, using only synthetic data")
            combined_df = synthetic_df
        else:
            # Combine datasets
            combined_df = pd.concat([real_df, synthetic_df], axis=0, ignore_index=True)
        
        # Clean the combined dataset
        combined_df = self.clean_dataframe(combined_df)
        
        # Engineer features
        if 'temperature' in combined_df.columns and 'humidity' in combined_df.columns:
            combined_df['temp_humidity_interaction'] = combined_df['temperature'] * combined_df['humidity']
            combined_df['environmental_risk'] = (
                (combined_df['temperature'] > 28).astype(int) +
                (combined_df['humidity'] > 75).astype(int)
            )
            if 'rainfall' in combined_df.columns:
                combined_df['environmental_risk'] += (combined_df['rainfall'] > 150).astype(int)
        
        # Encode categorical variables
        categorical_columns = combined_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            combined_df[col] = self.label_encoders[col].fit_transform(combined_df[col])
        
        # Scale numerical features
        numerical_columns = combined_df.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = numerical_columns.drop('outbreak_status')
        combined_df[numerical_columns] = self.scaler.fit_transform(combined_df[numerical_columns])
        
        # Final check for any remaining NaN values
        combined_df = combined_df.dropna()
        
        return combined_df

    def train_model(self):
        # Get real and synthetic data
        print("Fetching real data from database...")
        real_data = self.get_real_data()
        
        print("Generating synthetic data...")
        synthetic_data = self.generate_synthetic_data()
        
        # Preprocess and combine data
        combined_data = self.preprocess_combined_data(real_data, synthetic_data)
        
        # Verify no NaN values in the dataset
        if combined_data.isnull().any().any():
            raise ValueError("Dataset still contains NaN values after preprocessing")
        
        # Prepare features and target
        X = combined_data.drop('outbreak_status', axis=1)
        y = combined_data['outbreak_status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle class imbalance
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Train model with GridSearchCV
        print("Training Random Forest model...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        self.model = grid_search.best_estimator_
        return self.model, X_test, y_test

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance and generate visualizations"""
        y_pred = self.model.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Outbreak', 'Outbreak'],
                    yticklabels=['No Outbreak', 'Outbreak'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Plot ROC curve
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

def main():
    try:
        predictor = MalariaPredictionSystem()
        model, X_test, y_test = predictor.train_model()
        
        predictor.evaluate_model(X_test, y_test)
        
        with open('malaria_models.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("\nModel saved as 'malaria_models.pkl'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()