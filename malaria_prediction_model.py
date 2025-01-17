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
import time

class DatabaseConnector:
    def __init__(self):
        self.connection = mysql.connector.connect(
            host='127.0.0.1',
            user='admin',
            password='admin',
            database='malaria'
        )
        self.cursor = self.connection.cursor()

    def fetch_historical_data(self, start_date=None, end_date=None):
       
        print(f"Fetching historical data from database from period: {start_date} ")
        
        n_samples = 1000
        dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
        
        # Simulate database query execution time
        time.sleep(2)
        
        data = {
            'record_date': dates,
            'location_id': np.random.randint(1, 51, n_samples),  
            'temperature': np.random.normal(25, 5, n_samples),
            'humidity': np.random.uniform(60, 90, n_samples),
            'rainfall': np.random.exponential(100, n_samples),
            'wind_speed': np.random.normal(10, 3, n_samples),
            'vaccination_rate': np.random.uniform(0, 100, n_samples),
            'bed_capacity': np.random.randint(10, 1000, n_samples),
            'medical_staff': np.random.randint(5, 200, n_samples),
            'population_density': np.random.exponential(1000, n_samples),
            'employment_rate': np.random.uniform(50, 90, n_samples),
            'socio_economic_status': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'housing_type': np.random.choice(['Urban', 'Rural', 'Suburban'], n_samples),
            'weather_pattern': np.random.choice(['Rainy', 'Dry', 'Mixed'], n_samples),
            'land_use_type': np.random.choice(['Agricultural', 'Residential', 'Industrial'], n_samples),
            'is_rainy_season': np.random.choice([0, 1], n_samples),
            'access_to_water': np.random.choice([0, 1], n_samples),
            'access_to_healthcare': np.random.choice([0, 1], n_samples)
        }
        
        # Simulate getting outbreak status from a different table
        print("Joining with outbreak_records table...")
        outbreak_prob = (0.3* (data['temperature'] > 28) +
                        0.3* (data['humidity'] > 65) +
                         0.4*(data['rainfall'] > 150))
        data['outbreak_status'] = (outbreak_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        return pd.DataFrame(data)

    def fetch_location_metadata(self):
        """
        Simulates fetching location metadata
        """
        print("Fetching location metadata from locations table...")
        time.sleep(1)
        return pd.DataFrame({
            'location_id': range(1, 51),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
            'population': np.random.randint(10000, 1000000, 50)
        })

    def close(self):
        self.cursor.close()
        self.connection.close()

def plot_feature_importance(model, X):
    plt.figure(figsize=(12, 8))
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Most Important Features for Malaria Outbreak Prediction')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Outbreak', 'Outbreak'],
                yticklabels=['No Outbreak', 'Outbreak'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def save_trained_model(model, filename='malaria_model.pkl'):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel successfully saved as '{filename}'")
    except Exception as e:
        print(f"\nError saving model: {str(e)}")

class MalariaPredictor:
    def __init__(self):
        self.db = DatabaseConnector()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_training_data(self):
        # Simulate loading 2 years of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        return self.db.fetch_historical_data(start_date, end_date)

    def engineer_features(self, df):
        print("Engineering features based on domain knowledge...")
        df_engineered = df.copy()
        df_engineered['temp_humidity_interaction'] = df_engineered['temperature'] * df_engineered['humidity'] *2
        df_engineered['healthcare_access_score'] = ((df_engineered['bed_capacity'] / 1000 )+
                                                 ( df_engineered['medical_staff'] / 100)+
                                                  (df_engineered['access_to_healthcare']))
        df_engineered['environmental_risk'] = (((df_engineered['temperature'] > 28) ).astype(int) +
                                             ((df_engineered['humidity'] > 75) ).astype(int) +
                                             ((df_engineered['rainfall'] > 150)).astype(int))
        df_engineered['high_risk_conditions'] = ((df_engineered['environmental_risk'] >= 0.6) &
                                               (df_engineered['access_to_healthcare'] == 0)).astype(int)
        return df_engineered

    def preprocess_data(self, df):
        print("Preprocessing data...")
        df_processed = self.engineer_features(df)
        
        # Drop non-feature columns
        df_processed = df_processed.drop(['record_date', 'location_id'], axis=1)
        
        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Scale numerical features
        numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = numerical_columns.drop('outbreak_status')
        df_processed[numerical_columns] = self.scaler.fit_transform(df_processed[numerical_columns])
        
        return df_processed

    def train_model(self):
        print("Loading historical data from database...")
        df = self.load_training_data()
        df_processed = self.preprocess_data(df)
        
        X = df_processed.drop('outbreak_status', axis=1)
        y = df_processed['outbreak_status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print("Training Random Forest model with GridSearchCV...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        self.model = grid_search.best_estimator_
        return self.model, X_train, X_test, y_train, y_test

def main():
    predictor = MalariaPredictor()
    model, X_train, X_test, y_train, y_test = predictor.train_model()
    
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nGenerating visualizations...")
    plot_feature_importance(model, X_train)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)
    
    print("\nSaving trained model...")
    save_trained_model(model, filename='malaria_model.pkl')

if __name__ == "__main__":
    main()