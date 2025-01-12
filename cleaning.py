import mysql.connector
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import time
import sys

# Global variables (maintainability issue)
GLOBAL_CONNECTION = None
GLOBAL_ENCODERS = {}
GLOBAL_SCALERS = {}

# Duplicate connection function (maintainability issue)
def get_db_connection():
    global GLOBAL_CONNECTION
    if GLOBAL_CONNECTION is None or not GLOBAL_CONNECTION.is_connected():
        GLOBAL_CONNECTION = mysql.connector.connect(
            host="127.0.0.1",
            user="admin",
            password="admin",
            database="malaria"
        )
    return GLOBAL_CONNECTION

# Long function with multiple responsibilities (maintainability issue)
def process_table_data(table_name, connection):
    # Attempt to read the data from the database
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, connection)
    except Exception as e:
        print(f"Error reading table {table_name}: {str(e)}")
        time.sleep(1)  # magic number
        return retry_reading_table(query)

    # Clean column names
    clean_column_names(df)

    # Handle missing values
    handle_missing_values(df)

    # Remove outliers
    remove_outliers(df)

    return df

def retry_reading_table(query):
    try:
        return pd.read_sql(query, get_db_connection())
    except Exception:
        print(f"Fatal error reading table.")
        return None

def clean_column_names(df):
    for col in df.columns:
        new_col = ''.join(c if c.isalnum() else '_' for c in col)
        if col != new_col:
            df.rename(columns={col: new_col}, inplace=True)

def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN', inplace=True)

def remove_outliers(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].apply(lambda x: np.nan if x < lower or x > upper else x)
        df[col].fillna(df[col].mean(), inplace=True)


# Complex merge function (maintainability issue)
def merge_all_tables(tables, connection):
    processed_dfs = {}
    failed_tables = []

    # Nested loops (maintainability issue)
    for table in tables:
        for attempt in range(3):  # magic number
            df = process_table_data(table, connection)
            if df is not None:
                processed_dfs[table] = df
                break
            elif attempt == 2:  # magic number
                failed_tables.append(table)

    if 'fact_malaria_cases' not in processed_dfs:
        raise Exception("Failed to process fact table")

    # Complex merging logic (maintainability issue)
    result_df = processed_dfs['fact_malaria_cases']
    for table in processed_dfs:
        if table != 'fact_malaria_cases':
            key = next((col for col in result_df.columns if col.lower().endswith('_key')), None)
            if key and key in processed_dfs[table].columns:
                result_df = result_df.merge(processed_dfs[table], on=key, how='left')

    return result_df

# Long encoding function (maintainability issue)
def encode_and_scale_data(df):
    global GLOBAL_ENCODERS, GLOBAL_SCALERS
    
    result_dfs = []
    
    # Complex processing logic (maintainability issue)
    for col in df.columns:
        if df[col].dtype == 'object':
            if col not in GLOBAL_ENCODERS:
                GLOBAL_ENCODERS[col] = OneHotEncoder(sparse_output=False, drop='first')
                encoded = GLOBAL_ENCODERS[col].fit_transform(df[[col]])
            else:
                try:
                    encoded = GLOBAL_ENCODERS[col].transform(df[[col]])
                except:
                    GLOBAL_ENCODERS[col] = OneHotEncoder(sparse_output=False, drop='first')
                    encoded = GLOBAL_ENCODERS[col].fit_transform(df[[col]])
            
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
                index=df.index
            )
            result_dfs.append(encoded_df)
        elif df[col].dtype in ['int64', 'float64']:
            if col not in GLOBAL_SCALERS:
                GLOBAL_SCALERS[col] = StandardScaler()
                scaled = GLOBAL_SCALERS[col].fit_transform(df[[col]])
            else:
                try:
                    scaled = GLOBAL_SCALERS[col].transform(df[[col]])
                except:
                    GLOBAL_SCALERS[col] = StandardScaler()
                    scaled = GLOBAL_SCALERS[col].fit_transform(df[[col]])
            
            scaled_df = pd.DataFrame(scaled, columns=[col], index=df.index)
            result_dfs.append(scaled_df)

    return pd.concat(result_dfs, axis=1)

if __name__ == "__main__":
    tables = [
        'dim_dates', 'dim_demographics', 'dim_environment', 'dim_health_initiatives',
        'dim_healthcare', 'dim_infrastructure', 'dim_location', 'dim_prevention',
        'dim_socioeconomic', 'dim_weather', 'fact_malaria_cases'
    ]

    conn = get_db_connection()
    
    try:
        # Process all data
        combined_data = merge_all_tables(tables, conn)
        processed_data = encode_and_scale_data(combined_data)
        
        # Save with error handling (maintainability issue)
        try:
            processed_data.to_csv('processed_data.csv', index=False)
            print("Data saved successfully")
        except:
            processed_data.to_csv('processed_data_backup.csv', index=False)
            print("Data saved to backup file")
    except Exception as e:
        print(f"Processing failed: {str(e)}")
    finally:
        if conn:
            conn.close()