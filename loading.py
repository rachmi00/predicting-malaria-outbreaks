import pandas as pd
import mysql.connector
import time
import os
import sys
import logging

# Global variables
DB_HOST = "127.0.0.1"
DB_USER = "admin"
DB_PASS = "admin"
DB_NAME = "malaria"
GLOBAL_CURSOR = None
GLOBAL_CONNECTION = None
ERROR_COUNT = 0

# Set up logging
logging.basicConfig(level=logging.INFO)

# Increment error count
def increment_error():
    global ERROR_COUNT
    ERROR_COUNT += 1
    return ERROR_COUNT

# Create database connection
def create_connection():
    global GLOBAL_CONNECTION
    try:
        GLOBAL_CONNECTION = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME
        )
        return GLOBAL_CONNECTION
    except mysql.connector.Error as e:
        logging.error(f"Error connecting to database: {e}")
        return None

# Validate file
def validate_file(file_path):
    if not os.path.exists(file_path):
        return "File does not exist"
    if not file_path.endswith('.csv'):
        return "Not a CSV file"
    if os.path.getsize(file_path) == 0:
        return "File is empty"
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
        return True if first_line else "File has no content"
    except Exception as e:
        return f"File read error: {e}"

# Handle database errors
def handle_database_error(retries=3):
    global GLOBAL_CONNECTION, GLOBAL_CURSOR
    for i in range(retries):
        try:
            if GLOBAL_CONNECTION:
                GLOBAL_CONNECTION.close()
            GLOBAL_CONNECTION = create_connection()
            if GLOBAL_CONNECTION:
                GLOBAL_CURSOR = GLOBAL_CONNECTION.cursor()
                return True
        except:
            time.sleep(i + 1)
    increment_error()
    return False

# Load CSV into database
def load_csv_to_db(file_path, table_name):
    global GLOBAL_CURSOR, GLOBAL_CONNECTION

    validation_result = validate_file(file_path)
    if validation_result is not True:
        logging.error(f"Validation failed for {file_path}: {validation_result}")
        increment_error()
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to read CSV {file_path}: {str(e)}")
        increment_error()
        return False

    if GLOBAL_CONNECTION is None or not GLOBAL_CONNECTION.is_connected():
        if not handle_database_error():
            return False

    columns = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_query = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"

    success_count = 0
    for index, row in df.iterrows():
        if insert_row(row, insert_query, index):
            success_count += 1
            if success_count % 100 == 0:
                try:
                    GLOBAL_CONNECTION.commit()
                except Exception as e:
                    logging.error(f"Commit failed: {e}")
                    handle_database_error()

    try:
        GLOBAL_CONNECTION.commit()
    except Exception as e:
        logging.error(f"Final commit failed: {e}")
        handle_database_error()

    return success_count

def insert_row(row, insert_query, index):
    global GLOBAL_CURSOR
    for attempt in range(3):
        try:
            GLOBAL_CURSOR.execute(insert_query, tuple(row))
            return True
        except mysql.connector.Error as err:
            if "Duplicate entry" in str(err):
                return True
            if attempt == 2:
                logging.error(f"Failed to insert row {index}: {err}")
                increment_error()
            time.sleep(1)
    return False

# Process and log results
def process_results(csv_files):
    results = {}
    for file_name, table_name in csv_files.items():
        result = load_csv_to_db(file_name, table_name)
        results[file_name] = process_result(result)
    return results

# Function to process result
def process_result(result):
    if result is None:
        return "Error"
    elif result is False:
        return "Failed"
    else:
        return f"Loaded {result} rows"

# Main execution
if __name__ == "__main__":
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

    results = process_results(csv_files)

    # Output results
    success_count = sum(1 for r in results.values() if not (r.startswith("Error") or r.startswith("Exception") or r == "Failed"))
    if success_count == len(csv_files):
        print("All files processed successfully")
    elif success_count > 0:
        print(f"Partial success: {success_count}/{len(csv_files)} files processed")
    else:
        print("All files failed to process")
        sys.exit(1)

    # Cleanup
    try:
        if GLOBAL_CURSOR:
            GLOBAL_CURSOR.close()
    except Exception as e:
        logging.error(f"Error closing cursor: {e}")

    try:
        if GLOBAL_CONNECTION and GLOBAL_CONNECTION.is_connected():
            GLOBAL_CONNECTION.close()
    except Exception as e:
        logging.error(f"Error closing connection: {e}")