import pytest
import pandas as pd
import numpy as np
import mysql.connector
from unittest.mock import Mock, patch, MagicMock
from pandas.testing import assert_frame_equal
import cleaning  # The first script
import loading # The second script

# Test data fixtures
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'numeric_col': [1, 2, np.nan, 4, 5],
        'categorical_col': ['A', 'B', None, 'B', 'C'],
        'Date_key': [1, 2, 3, 4, 5]
    })

@pytest.fixture
def mock_db_connection():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn

# Preprocessing Script Tests
class TestPreprocessingFunctions:
    def test_connect_to_db(self, mock_db_connection):
        with patch('mysql.connector.connect', return_value=mock_db_connection):
            conn = cleaning.connect_to_db()
            assert conn is not None
            mysql.connector.connect.assert_called_once_with(
                host="127.0.0.1",
                user="admin",
                password="admin",
                database="malaria"
            )

    def test_fetch_table_data(self, mock_db_connection, sample_data):
        with patch('pandas.read_sql', return_value=sample_data):
            result = cleaning.fetch_table_data('test_table', mock_db_connection)
            assert_frame_equal(result, sample_data)
            pd.read_sql.assert_called_once()

    def test_handle_missing_values(self, sample_data):
        result = cleaning.handle_missing_values(sample_data)
        assert result['numeric_col'].isna().sum() == 0
        assert result['categorical_col'].isna().sum() == 0

    def test_remove_outliers(self):
        df = pd.DataFrame({
            'values': [1, 2, 3, 100, 4, 5]  # 100 is an outlier
        })
        result = cleaning.remove_outliers(df)
        assert 100 not in result['values'].values
        assert len(result) < len(df)

    def test_preprocess_table(self, mock_db_connection, sample_data):
        with patch('cleaning.fetch_table_data', return_value=sample_data):
            result = cleaning.preprocess_table('test_table', mock_db_connection)
            assert result is not None
            assert result['numeric_col'].isna().sum() == 0

    def test_combine_dataframes(self, mock_db_connection):
        mock_tables = {
            'fact_malaria_cases': pd.DataFrame({
                'Date_key': [1, 2],
                'cases': [10, 20]
            }),
            'dim_dates': pd.DataFrame({
                'Date_key': [1, 2],
                'date': ['2024-01-01', '2024-01-02']
            })
        }
        
        with patch('cleaning.preprocess_table', side_effect=lambda table, conn: mock_tables[table]):
            result = cleaning.combine_dataframes(['fact_malaria_cases', 'dim_dates'], mock_db_connection)
            assert 'date' in result.columns
            assert 'cases' in result.columns

    def test_encode_and_scale(self):
        input_df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        result = cleaning.encode_and_scale(input_df)
        assert result is not None
        assert result['numeric'].mean() == pytest.approx(0, abs=1e-10)
        assert result['numeric'].std() == pytest.approx(1, abs=1e-10)

# Data Loader Tests
class TestDataLoader:
    def test_load_csv_to_db(self, mock_db_connection):
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        with patch('pandas.read_csv', return_value=test_df), \
             patch('mysql.connector.connect', return_value=mock_db_connection):
            
            loading.load_csv_to_db('test.csv', 'test_table')
            
            # Verify cursor.execute was called for each row
            assert mock_db_connection.cursor().execute.call_count == 3
            mock_db_connection.commit.assert_called_once()

    def test_load_csv_to_db_handles_errors(self, mock_db_connection):
        test_df = pd.DataFrame({
            'col1': [1, 2],
            'col2': ['A', 'B']
        })
        
        mock_db_connection.cursor().execute.side_effect = [
            None,  # First row succeeds
            mysql.connector.Error('Test error')  # Second row fails
        ]
        
        with patch('pandas.read_csv', return_value=test_df), \
             patch('mysql.connector.connect', return_value=mock_db_connection):
            
            loading.load_csv_to_db('test.csv', 'test_table')
            
            # Verify the function continues after error
            assert mock_db_connection.cursor().execute.call_count == 2
            mock_db_connection.commit.assert_called_once()

    @patch('loading.load_csv_to_db')
    def test_main_script_processes_all_files(self, mock_load_csv):
        with patch('mysql.connector.connect', return_value=mock_db_connection):
            # Run the main script
            loading.main()
            
            # Verify load_csv_to_db was called for each table
            assert mock_load_csv.call_count == len(loading.csv_files)

# Integration test
def test_end_to_end_preprocessing(mock_db_connection, tmp_path):
    # Create test CSV file
    test_csv = tmp_path / "processed_data.csv"
    
    with patch('mysql.connector.connect', return_value=mock_db_connection), \
         patch('cleaning.combine_dataframes', return_value=pd.DataFrame({
             'numeric': [1, 2, 3],
             'categorical': ['A', 'B', 'C']
         })):
        
        cleaning.main()
        
        # Verify the processed data was saved
        assert test_csv.exists()

if __name__ == '__main__':
    pytest.main(['-v'])