import mysql.connector
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malaria_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    def __init__(self, host, user, password, database):
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database
        }
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            logger.info("Successfully connected to database")
            return self.connection
        except mysql.connector.Error as err:
            logger.error(f"Database connection failed: {err}")
            raise

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")

def get_warehouse_data(connection):
    """
    Fetch data from the data warehouse with error handling and data validation
    """
    try:
        query = """
        SELECT 
            CAST(fc.Outbreak_status AS UNSIGNED) as outbreak_status,
            dw.Temperature as temperature, 
            dw.Humidity as humidity, 
            dw.Rainfall as rainfall, 
            dw.Wind_speed as wind_speed, 
            dw.Weather_pattern as weather_pattern,
            dd.Vaccination_rate as vaccination_rate, 
            dd.Housing_type as housing_type, 
            dd.Socio_economic_status as socio_economic_status,
            dh.Bed_capacity as bed_capacity, 
            dh.Medical_staff as medical_staff,
            dl.Population_density as population_density,
            ds.Employment_rate as employment_rate,
            de.Land_use_type as land_use_type,
            CAST(dt.Is_rainy_season AS UNSIGNED) as is_rainy_season,
            CAST(di.Access_to_water AS UNSIGNED) as access_to_water,
            CAST(ds.Access_to_healthcare AS UNSIGNED) as access_to_healthcare
        FROM FACT_MALARIA_CASES fc
        JOIN DIM_WEATHER dw ON fc.Weather_key = dw.Weather_key
        JOIN DIM_DEMOGRAPHICS dd ON fc.Demographic_key = dd.Demographic_key
        JOIN DIM_HEALTHCARE dh ON fc.Healthcare_key = dh.Healthcare_key
        JOIN DIM_LOCATION dl ON fc.Location_key = dl.Location_key
        JOIN DIM_SOCIOECONOMIC ds ON fc.Socioeconomic_key = ds.Socioeconomic_key
        JOIN DIM_ENVIRONMENT de ON fc.Environment_key = de.Environment_key
        JOIN DIM_DATES dt ON fc.Date_key = dt.Date_key
        JOIN DIM_INFRASTRUCTURE di ON fc.Infrastructure_key = di.Infrastructure_key
        """
        
        df = pd.read_sql(query, connection)
        logger.info(f"Successfully fetched {len(df)} records from warehouse")
        
        # Validate data types and handle missing values
        numeric_columns = ['temperature', 'humidity', 'rainfall', 'wind_speed', 'vaccination_rate',
                         'bed_capacity', 'medical_staff', 'population_density', 'employment_rate']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with column medians
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching warehouse data: {str(e)}")
        raise

def create_sample_data(n_samples=1000):
    """
    Create synthetic data with consistent column names and data types
    """
    try:
        np.random.seed(42)
        temperature = np.random.normal(25, 5, n_samples)
        humidity = np.random.uniform(60, 90, n_samples)
        rainfall = np.random.exponential(100, n_samples)
        outbreak_prob = (0.3 * (temperature > 28) +
                        0.3 * (humidity > 75) +
                        0.4 * (rainfall > 150))
        outbreak_status = (outbreak_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        data = {
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': np.random.normal(10, 3, n_samples),
            'vaccination_rate': np.random.uniform(0, 60, n_samples),
            'bed_capacity': np.random.randint(10, 1000, n_samples),
            'medical_staff': np.random.randint(5, 200, n_samples),
            'population_density': np.random.exponential(1500, n_samples),
            'employment_rate': np.random.uniform(50, 90, n_samples),
            'socio_economic_status': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4,0.4,0.2]),
            'housing_type': np.random.choice(['Urban', 'Rural', 'Suburban'], n_samples, p=[0.5,0.3,0.2]),
            'weather_pattern': np.random.choice(['Rainy', 'Dry', 'Mixed'], n_samples, p=[0.6,0.1,0.3]),
            'land_use_type': np.random.choice(['Agricultural', 'Residential', 'Industrial'], n_samples),
            'is_rainy_season': np.random.choice([0, 1],n_samples, p=[0.2,0.8]),
            'access_to_water': np.random.choice([0, 1],n_samples, p=[0.2, 0.8]),
            'access_to_healthcare': np.random.choice([0, 1], n_samples, p=[0.8,0.2]),
            'outbreak_status': np.ones(n_samples, dtype=int) #All samples are outbreaks
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Successfully generated {n_samples} synthetic samples")
        return df
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise

def create_combined_dataset(connection, n_samples=1000):
    """
    Combine warehouse and synthetic data with validation
    """
    try:
        # Get both datasets
        warehouse_data = get_warehouse_data(connection)


        #calculate number of samples needed for better balance
        #synthetic_data = create_sample_data(n_samples)
        n_warehouse_outbreaks = warehouse_data['outbreak_status'].sum()
        n_warehouse_non_outbreaks = len(warehouse_data) - n_warehouse_outbreaks

          # Adjust synthetic samples to balance the dataset
        n_synthetic_needed = max(0, n_warehouse_non_outbreaks - n_warehouse_outbreaks)
        logger.info(f"Generating {n_synthetic_needed} synthetic outbreak samples to balance dataset")
        
        # Verify column alignment
        # warehouse_cols = set(warehouse_data.columns)
        # synthetic_cols = set(n_synthetic_needed.columns)
        
        if n_synthetic_needed > 0:
            synthetic_data = create_sample_data(n_samples =n_synthetic_needed)
          
        
        # Ensure consistent data types
            for col in warehouse_data.columns:
                if col in synthetic_data.columns:
                    target_dtype = synthetic_data[col].dtype
                    warehouse_data[col] = warehouse_data[col].astype(target_dtype)
        
        # Combine datasets
            combined_data = pd.concat([warehouse_data, synthetic_data], ignore_index=True)
        else:
            combined_data = warehouse_data
        # Validate combined dataset
        assert not combined_data.empty, "Combined dataset is empty"
        assert combined_data['outbreak_status'].isin([0, 1]).all(), "Invalid outbreak_status values"
        
        logger.info(f"Successfully created combined dataset with {len(combined_data)} records")
        return combined_data
        
    except Exception as e:
        logger.error(f"Error creating combined dataset: {str(e)}")
        raise

def engineer_features(df):
    """
    Engineer features with input validation and error handling
    """
    try:
        df_engineered = df.copy()
        
        # Validate required columns exist
        required_columns = ['temperature', 'humidity', 'rainfall', 'bed_capacity', 
                          'medical_staff', 'access_to_healthcare']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create features
               # 1. Temperature-based features
        df_engineered['temp_risk_score'] = (
            (df_engineered['temperature'] > 28).astype(int) * 3 +  # High weight for optimal malaria temperature
            (df_engineered['temperature'] > 32).astype(int) * 2 +  # Additional weight for very high temperatures
            (df_engineered['temperature'] < 20).astype(int) * -2   # Negative weight for low temperatures
        )
        
        # 2. Enhanced humidity interactions
        df_engineered['humidity_risk'] = (
            (df_engineered['humidity'] > 60).astype(int) * 1 +
            (df_engineered['humidity'] > 70).astype(int) * 1 +
            (df_engineered['humidity'] > 80).astype(int) * 2
        )
        
        # 3. Rainfall impact features
        df_engineered['rainfall_risk'] = (
            (df_engineered['rainfall'] > 50).astype(int) * 1 +
            (df_engineered['rainfall'] > 100).astype(int) * 2 +
            (df_engineered['rainfall'] > 150).astype(int) * 3
        )
        
        # 4. Complex weather interactions
        df_engineered['weather_interaction_score'] = (
            df_engineered['temp_risk_score'] * 
            df_engineered['humidity_risk'] * 
            (df_engineered['rainfall_risk'] + 1)
        )
        
        # 5. Seasonal weather pattern
        df_engineered['seasonal_risk'] = (
            df_engineered['is_rainy_season'].astype(int) * 2 +
            (df_engineered['weather_pattern'] == 'Rainy').astype(int) * 2
        )
        
        # 6. Wind impact (mosquito flight conditions)
        df_engineered['wind_impact'] = (
            (df_engineered['wind_speed'] < 15).astype(int) * 2 -  # Ideal conditions for mosquitoes
            (df_engineered['wind_speed'] > 25).astype(int) * 1    # Strong winds reduce mosquito activity
        )
        
        # Original features (modified with weather emphasis)
        df_engineered['temp_humidity_interaction'] = (
            df_engineered['temperature'] * 
            df_engineered['humidity'] * 
            1.5  # Increased weight
        )
        
        # Healthcare and environmental interaction
        df_engineered['healthcare_access_score'] = (
            df_engineered['bed_capacity'] / 100 +
            df_engineered['medical_staff'] / 20 +
            df_engineered['access_to_healthcare'] * 2
        ) * (1 + df_engineered['weather_interaction_score'] / 10)  # Modified by weather conditions
        
        # Overall environmental risk with weather emphasis
        df_engineered['environmental_risk'] = (
            df_engineered['temp_risk_score'] * 0.4 +
            df_engineered['humidity_risk'] * 0.3 +
            df_engineered['rainfall_risk'] * 0.3
        )
        
        # High risk conditions incorporating weather
        df_engineered['high_risk_conditions'] = (
            (df_engineered['environmental_risk'] >= 3) &
            (df_engineered['weather_interaction_score'] >= 5) &
            (df_engineered['access_to_healthcare'] == 0)
        ).astype(int)
        
        logger.info("Successfully engineered features")
        return df_engineered
        
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        raise

def preprocess_data(df):
    """
    Preprocess data with robust error handling
    """
    try:
        df_processed = engineer_features(df)
        
        # Identify column types
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = numerical_columns.drop('outbreak_status')
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in categorical_columns:
            df_processed[col] = le.fit_transform(df_processed[col])
        
        # Scale numerical variables
        scaler = StandardScaler()
        df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
        
        logger.info("Successfully preprocessed data")
        return df_processed
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def train_model(df):
    """
    Train model with cross-validation and balanced classes
    """
    try:
        X = df.drop('outbreak_status', axis=1)
        y = df['outbreak_status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Balance classes
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Train model
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_, X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def plot_feature_importance(model, X):
    """Plot feature importance with error handling"""
    try:
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
        
        # Save plot
        plt.savefig('feature_importance.png')
        logger.info("Feature importance plot saved as 'feature_importance.png'")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix with error handling"""
    try:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Outbreak', 'Outbreak'],
                    yticklabels=['No Outbreak', 'Outbreak'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('confusion_matrix.png')
        logger.info("Confusion matrix plot saved as 'confusion_matrix.png'")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve with error handling"""
    try:
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
        plt.tight_layout()
        
        # Save plot
        plt.savefig('roc_curve.png')
        logger.info("ROC curve plot saved as 'roc_curve.png'")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        raise

def save_trained_model(model, filename='malaria_model.pkl'):
    """Save trained model with error handling"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model successfully saved as '{filename}'")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_trained_model(filename='malaria_model.pkl'):
    """Load trained model with error handling"""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model successfully loaded from '{filename}'")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def validate_predictions(model, X_test, y_test):
    """Validate model predictions and generate performance metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        
        # Log results
        logger.info("\nModel Performance Metrics:")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nSensitivity (True Positive Rate): {sensitivity:.3f}")
        logger.info(f"Specificity (True Negative Rate): {specificity:.3f}")
        logger.info(f"Positive Predictive Value: {ppv:.3f}")
        logger.info(f"Negative Predictive Value: {npv:.3f}")
        
        return {
            'classification_report': report,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        }
        
    except Exception as e:
        logger.error(f"Error validating predictions: {str(e)}")
        raise

def main():
    """Main execution function with comprehensive error handling"""
    try:
        # Initialize database connector
        db = DatabaseConnector(
            host='127.0.0.1',
            user='admin',
            password='admin',
            database='malaria'
        )
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Connect to database
        connection = db.connect()
        
        try:
            logger.info("Starting malaria prediction model training pipeline...")
            
            # Create combined dataset
            logger.info("Creating combined dataset...")
            df = create_combined_dataset(connection)
            
            # Preprocess data
            logger.info("Preprocessing data...")
            df_processed = preprocess_data(df)
            
            # Train model
            logger.info("Training model...")
            model, X_train, X_test, y_train, y_test = train_model(df_processed)
            
            # Validate predictions
            logger.info("Validating model predictions...")
            validation_results = validate_predictions(model, X_test, y_test)
            
            # Create visualizations
            logger.info("Creating visualization plots...")
            plot_feature_importance(model, X_train)
            plot_confusion_matrix(y_test, model.predict(X_test))
            plot_roc_curve(model, X_test, y_test)
            
            # Save model with timestamp
            model_filename = f'malaria_model_{timestamp}.pkl'
            save_trained_model(model, filename=model_filename)
            
            # Save validation results
            results_filename = f'model_validation_results_{timestamp}.txt'
            with open(results_filename, 'w') as f:
                f.write(str(validation_results))
            
            logger.info("Model training pipeline completed successfully!")
            
        finally:
            # Ensure database connection is closed
            db.close()
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise