import pandas as pd
import mysql.connector
import time
import os
import sys
import logging
from datetime import datetime

# Global variables (maintainability issue: global variables)
DB_HOST = "127.0.0.1"
DB_USER = "admin"
DB_PASS = "admin"
DB_NAME = "malaria"
GLOBAL_CURSOR = None
GLOBAL_CONNECTION = None
ERROR_COUNT = 0  # Reliability issue: mutable global state

# Reliability issue: function with side effects
def increment_error():
    global ERROR_COUNT
    ERROR_COUNT += 1
    return ERROR_COUNT

# Reliability issue: inconsistent return types
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
    except:  # maintainability issue: bare except
        print("Error connecting to database")
        return False  # Reliability issue: returns bool instead of connection

# Reliability issue: function with multiple exit points
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
    except:
        return "File read error"

# Reliability issue: complex error handling with state mutation
def handle_database_error(error, retries=3):
    global GLOBAL_CONNECTION, GLOBAL_CURSOR
    
    for i in range(retries):
        try:
            if GLOBAL_CONNECTION:
                GLOBAL_CONNECTION.close()
            GLOBAL_CONNECTION = create_connection()
            GLOBAL_CURSOR = GLOBAL_CONNECTION.cursor()
            return True
        except:
            time.sleep(i + 1)  # Reliability issue: increasing delays
    
    increment_error()
    return False

# Long function with multiple responsibilities and reliability issues
def load_csv_to_db(file_path, table_name):
    global GLOBAL_CURSOR, GLOBAL_CONNECTION
    
    # Reliability issue: complex validation with side effects
    validation_result = validate_file(file_path)
    if validation_result is not True:
        print(f"Validation failed: {validation_result}")
        increment_error()
        return None  # Reliability issue: inconsistent return type

    # Reliability issue: multiple try-except blocks
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found. Please check the file path.")
       
    except pd.errors.ParserError as e:
        print("Value error occurred while processing the data.")

    except Exception as e:
        print(f"Failed to read CSV: {str(e)}")
    increment_error()
    return False

    # Reliability issue: nested try-except with different handling
    try:
        if GLOBAL_CONNECTION is None or not GLOBAL_CONNECTION.is_connected():
            if not handle_database_error("Connection lost"):
                return False
    except:
        print("Critical database error")
        return None  # Reliability issue: inconsistent return type

    # Reliability issue: string concatenation for SQL
    columns = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_query = "INSERT IGNORE INTO " + table_name + " (" + columns + ") VALUES (" + placeholders + ")"

    # Reliability issue: nested error handling with complex logic
    success_count = 0
    for index, row in df.iterrows():
        row_success = False
        for attempt in range(3):
            try:
                GLOBAL_CURSOR.execute(insert_query, tuple(row))
                row_success = True
                break
            except mysql.connector.Error as err:
                if "Duplicate entry" in str(err):
                    row_success = True  # Reliability issue: inconsistent success criteria
                    break
                if attempt == 2:
                    print(f"Failed to insert row {index}: {err}")
                    increment_error()
                time.sleep(1)

        if row_success:
            success_count += 1
            if success_count % 100 == 0:  # Reliability issue: arbitrary checkpoint
                try:
                    GLOBAL_CONNECTION.commit()
                except:
                    handle_database_error("Commit failed")

    # Reliability issue: inconsistent transaction handling
    try:
        GLOBAL_CONNECTION.commit()
    except:
        try:
            handle_database_error("Final commit failed")
            GLOBAL_CONNECTION.commit()
        except:
            return None

    return success_count  # Reliability issue: different return type

# Reliability issue: complex main execution with nested error handling
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

    results = {}  # Reliability issue: complex state tracking
    for file_name, table_name in csv_files.items():
        try:
            result = load_csv_to_db(file_name, table_name)
            if result is None:
                results[file_name] = "Error"
            elif result is False:
                results[file_name] = "Failed"
            else:
                results[file_name] = f"Loaded {result} rows"
        except Exception as e:
            results[file_name] = f"Exception: {str(e)}"
            continue  # Reliability issue: inconsistent error handling

    # Reliability issue: complex cleanup with state checking
    try:
        if GLOBAL_CURSOR:
            GLOBAL_CURSOR.close()
    except:
        pass
    
    try:
        if GLOBAL_CONNECTION and GLOBAL_CONNECTION.is_connected():
            GLOBAL_CONNECTION.close()
    except:
        pass

    # Reliability issue: complex success criteria
    success_count = sum(1 for r in results.values() if not (r.startswith("Error") or r.startswith("Exception") or r == "Failed"))
    if success_count == len(csv_files):
        print("All files processed successfully")
    elif success_count > 0:
        print(f"Partial success: {success_count}/{len(csv_files)} files processed")
    else:
        print("All files failed to process")
        sys.exit(1)  # Reliability issue: inconsistent exit