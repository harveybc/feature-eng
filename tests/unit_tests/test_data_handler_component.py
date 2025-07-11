"""
Unit tests for Data Handler Component

This module contains comprehensive unit tests for the Data Handler component,
validating all behavioral contracts defined in the design_unit.md specification.
Tests focus on behavioral requirements BR-DH-001, BR-DH-002, and BR-DH-003.

Author: Feature Engineering System
Date: 2025-07-10
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional

# Will import the data handler once refactored
# from app.data_handler import DataHandler, DataResult, ValidationResult, SaveResult


class MockDataResult:
    """Mock data loading result for testing purposes."""
    
    def __init__(self, is_valid: bool = True, data: pd.DataFrame = None, validation_errors: List[str] = None):
        self.is_valid = is_valid
        self.data = data if data is not None else pd.DataFrame()
        self.validation_errors = validation_errors or []


class MockValidationResult:
    """Mock validation result for testing purposes."""
    
    def __init__(self, is_valid: bool = True, missing_columns: List[str] = None, type_errors: Dict[str, str] = None):
        self.is_valid = is_valid
        self.missing_columns = missing_columns or []
        self.type_errors = type_errors or {}


class MockSaveResult:
    """Mock save result for testing purposes."""
    
    def __init__(self, success: bool = True, file_path: str = "", error_message: str = None):
        self.success = success
        self.file_path = file_path
        self.error_message = error_message


class TestDataHandlerComponentBehavior:
    """
    Test class for validating Data Handler Component behavioral contracts.
    
    This class tests the behavioral requirements:
    - BR-DH-001: CSV data loading with proper type inference and validation
    - BR-DH-002: Data format validation for required columns and data types
    - BR-DH-003: Data saving with format consistency and metadata preservation
    """

    # Test fixtures and setup
    @pytest.fixture
    def valid_csv_content(self):
        """
        Fixture providing valid CSV content with time-series data.
        
        Returns:
            str: Valid CSV content for testing
        """
        return """Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,104.0,1000
2023-01-02,104.0,108.0,103.0,107.0,1200"""

    @pytest.fixture
    def csv_with_missing_values(self):
        """
        Fixture providing CSV content with missing values.
        
        Returns:
            str: CSV content with missing values
        """
        return """Date,Open,High,Low,Close
2023-01-01,100.0,,99.0,104.0
2023-01-02,104.0,108.0,103.0,"""

    @pytest.fixture
    def csv_out_of_order(self):
        """
        Fixture providing CSV content with dates out of temporal order.
        
        Returns:
            str: CSV content with temporal ordering issues
        """
        return """Date,Open,High,Low,Close
2023-01-02,104.0,108.0,103.0,107.0
2023-01-01,100.0,105.0,99.0,104.0"""

    @pytest.fixture
    def csv_missing_columns(self):
        """
        Fixture providing CSV content missing required columns.
        
        Returns:
            str: CSV content missing Low and Close columns
        """
        return """Date,Open,High
2023-01-01,100.0,105.0"""

    @pytest.fixture
    def csv_invalid_types(self):
        """
        Fixture providing CSV content with invalid data types.
        
        Returns:
            str: CSV content with type validation issues
        """
        return """Date,Open,High,Low,Close
invalid_date,not_number,105.0,99.0,104.0"""

    @pytest.fixture
    def sample_feature_data(self):
        """
        Fixture providing sample processed feature data.
        
        Returns:
            pd.DataFrame: Sample feature data for testing save operations
        """
        return pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Open': [100.0, 104.0, 108.0],
            'High': [105.0, 108.0, 112.0],
            'Low': [99.0, 103.0, 107.0],
            'Close': [104.0, 107.0, 111.0],
            'Volume': [1000, 1200, 1100],
            'SMA': [102.0, 105.5, 109.0],
            'RSI': [50.0, 55.0, 60.0]
        })

    # BR-DH-001: CSV Data Loading Tests
    def test_br_dh_001_loads_valid_csv_data(self, valid_csv_content):
        """
        Verify that data handler correctly loads well-formed CSV data
        with proper type inference and structure validation.
        
        Behavioral Contract: BR-DH-001
        Test ID: UT-DH-001
        """
        # Given: Data handler and valid CSV content
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Loading CSV data
        result = data_handler.load_csv(valid_csv_content)
        
        # Then: Data is correctly loaded and structured
        assert result.is_valid == True
        assert result.data.shape == (2, 6)
        assert result.data.columns.tolist() == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        assert result.data['Open'].dtype in ['float64', 'float32']
        assert result.data['Volume'].dtype in ['int64', 'int32']
        
    def test_br_dh_001_handles_missing_values_appropriately(self, csv_with_missing_values):
        """
        Verify that data handler handles missing values according
        to configured strategies (fill, drop, or error).
        
        Behavioral Contract: BR-DH-001
        Test ID: UT-DH-002
        """
        # Given: Data handler and CSV data with missing values
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Loading with fill strategy
        result = data_handler.load_csv(csv_with_missing_values, missing_strategy='fill')
        
        # Then: Missing values are handled appropriately
        assert result.is_valid == True
        assert not result.data.isnull().any().any()
        
        # Verify that missing values were filled
        assert result.data.loc[0, 'High'] != ""  # Should be filled
        assert result.data.loc[1, 'Close'] != ""  # Should be filled
        
    def test_br_dh_001_handles_missing_values_with_drop_strategy(self, csv_with_missing_values):
        """
        Verify that data handler correctly drops rows with missing values
        when using the drop strategy.
        
        Behavioral Contract: BR-DH-001
        Test ID: UT-DH-003
        """
        # Given: Data handler and CSV data with missing values
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Loading with drop strategy
        result = data_handler.load_csv(csv_with_missing_values, missing_strategy='drop')
        
        # Then: Rows with missing values are dropped
        assert result.is_valid == True
        assert result.data.shape[0] == 0  # All rows had missing values, so all dropped
        
    def test_br_dh_001_handles_missing_values_with_error_strategy(self, csv_with_missing_values):
        """
        Verify that data handler reports errors when encountering missing values
        with the error strategy.
        
        Behavioral Contract: BR-DH-001
        Test ID: UT-DH-004
        """
        # Given: Data handler and CSV data with missing values
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Loading with error strategy
        result = data_handler.load_csv(csv_with_missing_values, missing_strategy='error')
        
        # Then: Missing values cause validation errors
        assert result.is_valid == False
        assert len(result.validation_errors) > 0
        assert any('missing values' in error.lower() for error in result.validation_errors)
        
    def test_br_dh_001_validates_time_series_requirements(self, csv_out_of_order):
        """
        Verify that data handler validates time-series specific
        requirements like temporal ordering and consistency.
        
        Behavioral Contract: BR-DH-001
        Test ID: UT-DH-005
        """
        # Given: Data handler and CSV data with invalid time series structure
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Loading with time series validation
        result = data_handler.load_csv(csv_out_of_order, validate_time_series=True)
        
        # Then: Time series validation errors are reported
        assert result.is_valid == False
        assert len(result.validation_errors) > 0
        assert any('temporal order' in error.lower() for error in result.validation_errors)
        
    def test_br_dh_001_infers_data_types_correctly(self, valid_csv_content):
        """
        Verify that data handler correctly infers data types for
        different column types (numeric, temporal, categorical).
        
        Behavioral Contract: BR-DH-001
        Test ID: UT-DH-006
        """
        # Given: Data handler and CSV with mixed data types
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Loading CSV with type inference
        result = data_handler.load_csv(valid_csv_content, infer_types=True)
        
        # Then: Data types are correctly inferred
        assert result.is_valid == True
        assert pd.api.types.is_numeric_dtype(result.data['Open'])
        assert pd.api.types.is_numeric_dtype(result.data['Volume'])
        # Date column handling depends on implementation

    # BR-DH-002: Data Format Validation Tests
    def test_br_dh_002_validates_required_columns(self, csv_missing_columns):
        """
        Verify that data handler validates presence of required columns
        for time-series feature engineering operations.
        
        Behavioral Contract: BR-DH-002
        Test ID: UT-DH-007
        """
        # Given: Data handler and CSV data missing required columns
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Validating data format
        result = data_handler.validate_format(csv_missing_columns)
        
        # Then: Missing column errors are reported
        assert result.is_valid == False
        assert 'Low' in result.missing_columns
        assert 'Close' in result.missing_columns
        
    def test_br_dh_002_validates_data_types(self, csv_invalid_types):
        """
        Verify that data handler validates data types for numeric
        and temporal columns according to requirements.
        
        Behavioral Contract: BR-DH-002
        Test ID: UT-DH-008
        """
        # Given: Data handler and CSV data with invalid data types
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Validating data types
        result = data_handler.validate_format(csv_invalid_types)
        
        # Then: Type validation errors are reported
        assert result.is_valid == False
        assert 'Date' in result.type_errors
        assert 'Open' in result.type_errors
        
    def test_br_dh_002_validates_numeric_column_ranges(self):
        """
        Verify that data handler validates numeric columns are within
        reasonable ranges for financial data.
        
        Behavioral Contract: BR-DH-002
        Test ID: UT-DH-009
        """
        # Given: Data handler and CSV with values out of reasonable range
        invalid_range_csv = """Date,Open,High,Low,Close,Volume
2023-01-01,-100.0,105.0,99.0,104.0,1000
2023-01-02,104.0,50.0,150.0,107.0,-500"""  # High < Low, negative values
        
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Validating data format with range checks
        result = data_handler.validate_format(invalid_range_csv, validate_ranges=True)
        
        # Then: Range validation errors are reported
        assert result.is_valid == False
        assert len(result.validation_errors) > 0
        
    def test_br_dh_002_validates_column_consistency(self):
        """
        Verify that data handler validates logical consistency between
        related columns (e.g., High >= Low, Volume >= 0).
        
        Behavioral Contract: BR-DH-002
        Test ID: UT-DH-010
        """
        # Given: Data handler and CSV with logical inconsistencies
        inconsistent_csv = """Date,Open,High,Low,Close,Volume
2023-01-01,100.0,95.0,105.0,104.0,1000"""  # High < Low
        
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # When: Validating logical consistency
        result = data_handler.validate_format(inconsistent_csv, validate_consistency=True)
        
        # Then: Consistency validation errors are reported
        assert result.is_valid == False
        assert any('High' in error and 'Low' in error for error in result.validation_errors)

    # BR-DH-003: Data Saving Tests
    def test_br_dh_003_saves_data_with_correct_format(self, sample_feature_data):
        """
        Verify that data handler saves processed data maintaining
        format consistency and metadata preservation.
        
        Behavioral Contract: BR-DH-003
        Test ID: UT-DH-011
        """
        # Given: Data handler and processed feature data
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        output_path = 'test_output.csv'
        
        # When: Saving data
        result = data_handler.save_csv(sample_feature_data, output_path)
        
        # Then: Data is saved with correct format
        assert result.success == True
        assert result.file_path == output_path
        
        # Verify saved content can be loaded back
        loaded_result = data_handler.load_csv_from_file(output_path)
        assert loaded_result.is_valid == True
        pd.testing.assert_frame_equal(loaded_result.data, sample_feature_data)
        
    def test_br_dh_003_preserves_data_types_during_save(self, sample_feature_data):
        """
        Verify that data handler preserves data types when saving
        and loading data files.
        
        Behavioral Contract: BR-DH-003
        Test ID: UT-DH-012
        """
        # Given: Data handler and feature data with specific types
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        output_path = 'test_types.csv'
        
        # Ensure specific data types
        sample_feature_data['Volume'] = sample_feature_data['Volume'].astype('int64')
        sample_feature_data['SMA'] = sample_feature_data['SMA'].astype('float64')
        
        # When: Saving and reloading data
        save_result = data_handler.save_csv(sample_feature_data, output_path, preserve_types=True)
        load_result = data_handler.load_csv_from_file(output_path, infer_types=True)
        
        # Then: Data types are preserved
        assert save_result.success == True
        assert load_result.is_valid == True
        assert load_result.data['Volume'].dtype == sample_feature_data['Volume'].dtype
        assert load_result.data['SMA'].dtype == sample_feature_data['SMA'].dtype
        
    def test_br_dh_003_saves_with_metadata_preservation(self, sample_feature_data):
        """
        Verify that data handler saves data with metadata preservation
        including column descriptions and processing history.
        
        Behavioral Contract: BR-DH-003
        Test ID: UT-DH-013
        """
        # Given: Data handler and feature data with metadata
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        output_path = 'test_metadata.csv'
        
        # Add metadata to the data
        metadata = {
            'SMA': 'Simple Moving Average (20 period)',
            'RSI': 'Relative Strength Index',
            'processing_timestamp': datetime.now().isoformat(),
            'source_file': 'original_data.csv'
        }
        
        # When: Saving data with metadata
        result = data_handler.save_csv(sample_feature_data, output_path, metadata=metadata)
        
        # Then: Data is saved with metadata preserved
        assert result.success == True
        assert result.metadata_preserved == True
        
    def test_br_dh_003_handles_save_errors_gracefully(self, sample_feature_data):
        """
        Verify that data handler handles save errors gracefully
        with proper error reporting and cleanup.
        
        Behavioral Contract: BR-DH-003
        Test ID: UT-DH-014
        """
        # Given: Data handler and invalid save path
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        invalid_path = '/invalid/path/that/does/not/exist/file.csv'
        
        # When: Attempting to save to invalid path
        result = data_handler.save_csv(sample_feature_data, invalid_path)
        
        # Then: Save error is handled gracefully
        assert result.success == False
        assert result.error_message is not None
        assert 'path' in result.error_message.lower() or 'file' in result.error_message.lower()

    # Performance and Integration Tests
    def test_data_handler_performance_with_large_datasets(self):
        """
        Verify that data handler performs efficiently with large datasets
        without excessive memory usage or processing time.
        
        Test ID: UT-DH-015
        """
        # Given: Data handler and large dataset
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        # Create large dataset (10,000 rows)
        large_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10000, freq='H'),
            'Open': np.random.uniform(100, 200, 10000),
            'High': np.random.uniform(200, 300, 10000),
            'Low': np.random.uniform(50, 100, 10000),
            'Close': np.random.uniform(100, 200, 10000),
            'Volume': np.random.randint(1000, 10000, 10000)
        })
        
        import time
        start_time = time.time()
        
        # When: Processing large dataset
        csv_content = large_data.to_csv(index=False)
        result = data_handler.load_csv(csv_content)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Then: Performance requirements are met
        assert result.is_valid == True
        assert processing_time < 10.0  # Should process in under 10 seconds
        assert result.data.shape[0] == 10000
        
    def test_data_handler_handles_edge_cases_safely(self):
        """
        Verify that data handler handles edge cases safely
        including empty files, single row data, and malformed CSV.
        
        Test ID: UT-DH-016
        """
        # Given: Data handler and various edge cases
        from app.data_handler import DataHandler
        data_handler = DataHandler()
        
        edge_cases = [
            "",  # Empty file
            "Date,Open,High,Low,Close",  # Headers only
            "2023-01-01,100.0,105.0,99.0,104.0",  # Single row, no headers
            "Date,Open,High,Low,Close\n2023-01-01,100.0,105.0,99.0,104.0\nmalformed line"  # Malformed data
        ]
        
        # When: Processing edge cases
        for i, edge_case in enumerate(edge_cases):
            result = data_handler.load_csv(edge_case, handle_errors=True)
            
            # Then: Edge cases are handled safely without crashes
            assert result is not None
            assert hasattr(result, 'is_valid')
            # Each case may be valid or invalid, but should not crash


# Helper functions for creating test data
def create_sample_feature_data():
    """Create sample feature data for testing."""
    return pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Open': [100.0, 104.0, 108.0],
        'High': [105.0, 108.0, 112.0],
        'Low': [99.0, 103.0, 107.0],
        'Close': [104.0, 107.0, 111.0],
        'Volume': [1000, 1200, 1100],
        'SMA': [102.0, 105.5, 109.0],
        'RSI': [50.0, 55.0, 60.0]
    })


if __name__ == '__main__':
    # Run the tests when script is executed directly
    pytest.main([__file__, '-v', '--tb=short'])
