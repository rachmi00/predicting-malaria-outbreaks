import mysql.connector
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Database connection
def connect_to_db():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="admin",
        password="admin",
        database="malaria"
    )

# Fetch data from a specific table
def fetch_table_data(table_name, conn):
    print(f"Fetching data from {table_name}...")
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, conn)

# Handle missing values
def handle_missing_values(df):
    print("Handling missing values...")
    return df.fillna(0)

# Remove outliers using IQR method
def remove_outliers(df):
    print("Removing outliers...")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Preprocess a table
def preprocess_table(table_name, conn):
    df = fetch_table_data(table_name, conn)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    return df

# Combine fact table with dimension tables
def combine_dataframes(tables, conn):
    print("\nCombining data from all tables...")
    dfs = {table: preprocess_table(table, conn) for table in tables}

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
            print(f"Merging {dim_table} with fact_malaria_cases on {key}...")
            merged_df = merged_df.merge(dfs[dim_table], on=key, how='left')

    return merged_df

# Encode categorical and scale numerical data
def encode_and_scale(df):
    print("\nEncoding categorical data and scaling numerical data...")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Encode categorical data
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_categories = pd.DataFrame(
        encoder.fit_transform(df[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index
    )

    # Scale numerical data
    scaler = StandardScaler()
    scaled_numerics = pd.DataFrame(
        scaler.fit_transform(df[numerical_columns]),
        columns=numerical_columns,
        index=df.index
    )

    # Combine processed data
    processed_df = pd.concat([scaled_numerics, encoded_categories], axis=1)
    print("Encoding and scaling complete.")
    return processed_df

# Main processing
if __name__ == "_main_":
    conn = connect_to_db()

    tables = [
        'dim_dates', 'dim_demographics', 'dim_environment', 'dim_health_initiatives',
        'dim_healthcare', 'dim_infrastructure', 'dim_location', 'dim_prevention',
        'dim_socioeconomic', 'dim_weather', 'fact_malaria_cases'
    ]

    # Combine all data
    combined_data = combine_dataframes(tables, conn)

    # Encode and scale the data
    preprocessed_data = encode_and_scale(combined_data)

    # Save the processed data for machine learning
    preprocessed_data.to_csv('processed_data.csv', index=False)
    print("\nData preprocessing complete. Saved processed data to 'processed_data.csv'.")

    conn.close()