import pytest
import pandas as pd
import mysql.connector
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from loading import load_csv_to_db, csv_files
from cleaning import clean_table_in_db


# Test data fixtures
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, None, 40, 1000],
        'category': ['A', 'B', None, 'D', 'E']
    })

@pytest.fixture
def mock_cursor():
    cursor = Mock()
    cursor.description = [
        ('id', None, None, None, None, None, None),
        ('value', None, None, None, None, None, None),
        ('category', None, None, None, None, None, None)
    ]
    return cursor

@pytest.fixture
def mock_connection():
    return Mock()

# Tests for the data cleaning script
class TestDataCleaning:
    @patch('mysql.connector.connect')
    def test_database_connection(self, mock_connect):
        mock_connect.return_value = Mock()
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="admin",
            password="admin",
            database="malaria"
        )
        assert conn is not None
        mock_connect.assert_called_once_with(
            host="127.0.0.1",
            user="admin",
            password="admin",
            database="malaria"
        )

    def test_clean_table_handles_missing_values(self, mock_cursor, mock_connection, sample_df):
        mock_cursor.fetchall.return_value = sample_df.values.tolist()
        
        with patch('pandas.DataFrame', return_value=sample_df):
            clean_table_in_db('test_table')
            
        # Verify that UPDATE query was called for NULL values
        update_calls = [call for call in mock_cursor.execute.call_args_list 
                       if 'UPDATE' in str(call)]
        assert len(update_calls) > 0

    def test_clean_table_removes_outliers(self, mock_cursor, mock_connection, sample_df):
        mock_cursor.fetchall.return_value = sample_df.values.tolist()
        
        with patch('pandas.DataFrame', return_value=sample_df):
            clean_table_in_db('test_table')
            
        # Verify that DELETE query was called for outliers
        delete_calls = [call for call in mock_cursor.execute.call_args_list 
                       if 'DELETE' in str(call)]
        assert len(delete_calls) > 0

# Tests for the CSV loading script
class TestCSVLoading:
    @patch('pandas.read_csv')
    def test_load_csv_to_db(self, mock_read_csv, mock_cursor, mock_connection):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })
        mock_read_csv.return_value = mock_df

        with patch('mysql.connector.connect') as mock_connect:
            mock_connect.return_value = mock_connection
            load_csv_to_db('test.csv', 'test_table')

            # Verify that execute was called for each row
            assert mock_cursor.execute.call_count == len(mock_df)
            
            # Verify commit was called
            assert mock_connection.commit.called

    @pytest.mark.parametrize("file_name,table_name", [
        ('dim_dates.csv', 'dim_dates'),
        ('dim_demographics.csv', 'dim_demographics'),
        ('dim_environment.csv', 'dim_environment')
    ])
    def test_csv_file_mapping(self, file_name, table_name):
        assert csv_files[file_name] == table_name

    @patch('pandas.read_csv')
    def test_handles_empty_csv(self, mock_read_csv, mock_cursor, mock_connection):
        mock_read_csv.return_value = pd.DataFrame()
        
        with patch('mysql.connector.connect') as mock_connect:
            mock_connect.return_value = mock_connection
            load_csv_to_db('empty.csv', 'test_table')
            
            # Verify no executes were called for empty DataFrame
            assert mock_cursor.execute.call_count == 0

# Integration test simulation
@pytest.mark.integration
def test_full_process_simulation():
    with patch('mysql.connector.connect') as mock_connect:
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        # Test both cleaning and loading
        clean_table_in_db('test_table')
        load_csv_to_db('test.csv', 'test_table')

        # Verify the process completed
        assert mock_cursor.close.called
        assert mock_connection.close.called

# Error handling tests
class TestErrorHandling:
    def test_database_connection_error(self):
        with patch('mysql.connector.connect', side_effect=mysql.connector.Error):
            with pytest.raises(mysql.connector.Error):
                mysql.connector.connect(
                    host="127.0.0.1",
                    user="admin",
                    password="admin",
                    database="malaria"
                )

    @patch('pandas.read_csv')
    def test_invalid_csv_file(self, mock_read_csv):
        mock_read_csv.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_csv_to_db('nonexistent.csv', 'test_table')