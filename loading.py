import pandas as pd
import mysql.connector

# Step 1: Establish a database connection
db_connection = mysql.connector.connect(
    host="127.0.0.1",
    user="admin",  
    password="admin",  
    database="malaria"
)

cursor = db_connection.cursor()

# Step 3: Function to create an insert query from a DataFrame
def load_csv_to_db(file_path, table_name):
    # Read CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Dynamically build the SQL INSERT IGNORE query
    columns = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_query = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    # Insert each row into the table
    for row in df.itertuples(index=False):
        try:
            cursor.execute(insert_query, tuple(row))
        except mysql.connector.Error as err:
            print(f"Error inserting row into {table_name}: {err}")
            continue
    
    # Commit the transaction
    db_connection.commit()
    print(f"Data from {file_path} inserted into {table_name} table.")

# Step 4: Define file paths and table names
csv_files = {
    'dim_dates.csv': 'dim_dates',
    'dim_demographics.csv': 'dim_demographics',
    'dim_environment.csv': 'dim_environment',
    'dim_health_initiatives.csv': 'dim_health_initiatives',
    'dim_healthcare.csv': 'dim_healthcare',
    'dim_infrastructure.csv': 'dim_infrastructure',
    'dim_location.csv': 'dim_location',
    'dim_prevention.csv': 'dim_prevention',
    'dim_socioeconomic.csv': 'dim_socioeconomic',
    'dim_weather.csv': 'dim_weather',
    'fact_malaria_cases.csv': 'fact_malaria_cases'
}

# Step 5: Load each CSV file into the database
for file_name, table_name in csv_files.items():
    load_csv_to_db(file_name, table_name)

# Close the cursor and connection
cursor.close()
db_connection.close()