import mysql.connector
import pandas as pd

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="127.0.0.1", 
    user="admin",  
    password="admin",  
    database="malaria"  
)

# Create a cursor object to interact with the database
cursor = conn.cursor()

# List of tables to process
tables = [
    'dim_dates',
    'dim_demographics',
    'dim_environment',
    'dim_health_initiatives',
    'dim_healthcare',
    'dim_infrastructure',
    'dim_location',
    'dim_prevention',
    'dim_socioeconomic',
    'dim_weather',
    'fact_malaria_cases'
]

# Function to handle missing values and remove outliers directly in the database
def clean_table_in_db(table_name):
    print(f"\nFetching data from {table_name}...")
    # Load data from the table into a DataFrame
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    table_data = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])

    # Step 1: Check for Missing Values
    print(f"Checking for missing values in {table_name}...")
    missing_values = table_data.isnull().sum()
    print(f"Missing values in {table_name}:")
    print(missing_values)

    # Fill missing values with 0 or any appropriate method
    table_data.fillna(0, inplace=True)

    # Update missing values directly in the database
    for column in missing_values[missing_values > 0].index:
        update_query = f"""
        UPDATE {table_name}
        SET {column} = 0
        WHERE {column} IS NULL
        """
        cursor.execute(update_query)

    # Step 2: Remove Outliers using IQR method for numerical columns
    print(f"Handling outliers in {table_name}...")
    numerical_columns = table_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        Q1 = table_data[col].quantile(0.25)
        Q3 = table_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove rows where the value is outside the IQR bounds
        outliers_query = f"""
        DELETE FROM {table_name}
        WHERE {col} < {lower_bound} OR {col} > {upper_bound}
        """
        cursor.execute(outliers_query)

    # Step 3: Verify the changes
    print(f"Data after handling missing values and outliers in {table_name}:")
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    print(cursor.fetchall())

# Clean each table directly in the database
for table in tables:
    clean_table_in_db(table)

# Commit changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("\nAll tables cleaned successfully in the database!")
