import numpy as np
import pytest
import tempfile
import csv
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Dict, Any

from myclt.ML.supervised_learning.linear_regression.data import (
    Dataset, Prepareddata, load_csv_dataset, manual_input_dataset, select_features_and_target
)


class TestLoadCSVDataset:
    """Test cases for load_csv_dataset function."""
    
    def test_load_valid_csv_comma_delimited(self, tmp_path):
        """Test loading a valid CSV file with comma delimiter."""
        # Create a temporary CSV file with comma delimiter
        csv_path = tmp_path / "test_comma.csv"
        csv_content = """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9"""
        
        csv_path.write_text(csv_content)
        
        # Load dataset
        dataset = load_csv_dataset(str(csv_path))
        
        # Verify dataset properties
        assert isinstance(dataset, Dataset)
        assert dataset.data.shape == (5, 4)  # 5 rows, 4 columns
        assert dataset.columns == ["TV", "Radio", "Newspaper", "Sales"]
        assert np.all(np.isfinite(dataset.data))  # No NaN or inf values
        
        # Verify data values (spot check first row)
        np.testing.assert_allclose(dataset.data[0], [230.1, 37.8, 69.2, 22.1])
        np.testing.assert_allclose(dataset.data[-1], [180.8, 10.8, 58.4, 12.9])

    def test_load_valid_csv_semicolon_delimited(self, tmp_path):
        """Test loading a valid CSV file with semicolon delimiter."""
        # Create a temporary CSV file with semicolon delimiter
        csv_path = tmp_path / "test_semicolon.csv"
        csv_content = """house_age;distance_to_mrt;num_convenience_stores;house_price
32.0;84.87882;10;37.9
19.5;306.5947;9;42.2
13.3;561.9845;5;47.3
13.3;561.9845;5;54.8
5.0;390.5684;5;43.1"""
        
        csv_path.write_text(csv_content)
        
        # Load dataset
        dataset = load_csv_dataset(str(csv_path))
        
        # Verify dataset properties
        assert isinstance(dataset, Dataset)
        assert dataset.data.shape == (5, 4)  # 5 rows, 4 columns
        assert dataset.columns == ["house_age", "distance_to_mrt", "num_convenience_stores", "house_price"]
        assert np.all(np.isfinite(dataset.data))

    def test_load_csv_with_missing_values_raises_error(self, tmp_path):
        """Test that CSV with missing values raises ValueError."""
        csv_path = tmp_path / "test_missing.csv"
        csv_content = """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,,45.1,10.4
17.2,45.9,,9.3
151.5,41.3,58.5,18.5"""
        
        csv_path.write_text(csv_content)
        
        with pytest.raises(ValueError, match="Empty value"):
            load_csv_dataset(str(csv_path))

    def test_load_csv_with_row_column_mismatch_raises_error(self, tmp_path):
        """Test that CSV with inconsistent column count raises ValueError."""
        csv_path = tmp_path / "test_mismatch.csv"
        csv_content = """TV,Radio,Newspaper,Sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4,999.9  # Extra column
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5"""
        
        csv_path.write_text(csv_content)
        
        with pytest.raises(ValueError, match=r"has \d+ columns, expected"):
            load_csv_dataset(str(csv_path))

    def test_load_non_existent_file_raises_filenotfounderror(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_csv_dataset("/path/to/nonexistent/file.csv")

    def test_load_csv_with_numeric_and_nonnumeric_mixed_case(self, tmp_path):
        """Test CSV with both numeric and non-numeric columns."""
        csv_path = tmp_path / "test_mixed.csv"
        csv_content = """Age,Income,Category,Spending
25,50000,A,1200
32,75000,B,1500
45,60000,A,1100
28,40000,C,900
50,90000,B,1800"""
        
        csv_path.write_text(csv_content)
        
        # For regression, mixed types are handled but may result in some columns not being numeric
        # Let's check that the dataset is loaded successfully
        dataset = load_csv_dataset(str(csv_path))
        
        assert isinstance(dataset, Dataset)
        assert dataset.data.shape == (5, 4)
        assert dataset.columns == ["Age", "Income", "Category", "Spending"]
        # Mixed datasets are allowed, but feature validation happens during feature selection


class TestManualInputDataset:
    """Test cases for manual_input_dataset function."""
    
    def test_manual_input_valid_numeric_data(self):
        """Test manual input with valid numeric data."""
        # Mock user input for column names and data
        input_sequence = [
            "Feature1;Target",  # Column names (need at least 2 columns)
            "1.0;10.0",    # Row 1
            "3.0;20.0",    # Row 2  
            "5.0;30.0",    # Row 3
            "7.0;40.0",    # Row 4
            "9.0;50.0",   # Row 5
            ""             # Empty line to finish
        ]
        
        with patch("builtins.input", side_effect=input_sequence):
            dataset = manual_input_dataset()
        
        # Verify dataset properties - need at least 2 columns
        assert isinstance(dataset, Dataset)
        assert dataset.data.shape == (5, 2)
        assert dataset.columns == ["Feature1", "Target"]
        np.testing.assert_allclose(dataset.data[0], [1.0, 10.0])
        np.testing.assert_allclose(dataset.data[-1], [9.0, 50.0])

    def test_manual_input_less_than_5_rows_raises_error(self):
        """Test that providing less than 5 rows raises ValueError."""
        input_sequence = [
            "Feature1;Target",  # Column names
            "1.0;10.0",    # Row 1
            "3.0;20.0",    # Row 2  
            ""             # Empty line - only 2 rows
        ]
        
        with patch("builtins.input", side_effect=input_sequence):
            with pytest.raises(ValueError, match="at least 5 rows"):
                manual_input_dataset()

    def test_manual_input_less_than_2_columns_raises_error(self):
        """Test that providing less than 2 columns raises ValueError."""
        input_sequence = [
            "1",  # Only 1 column
            "Feature1",  # Single column name
        ]
        
        with patch("builtins.input", side_effect=input_sequence):
            with pytest.raises(ValueError, match="need at least 2 columns"):
                manual_input_dataset()

    def test_manual_input_non_numeric_values_raises_error(self):
        """Test that non-numeric values in data raise ValueError."""
        input_sequence = [
            "Feature1;Target",  # Column names
            "1.0;10.0",        # Row 1
            "3.0;20.0",        # Row 2  
            "5.0;30.0",        # Row 3
            "7.0;40.0",        # Row 4
            "abc;50.0",        # Row 5 with non-numeric value (causes error)
            "",                # After error, function asks for same row again
        ]
        
        # Just test that it raises a specific error - don't test the exact message
        # as the function might ask for input multiple times
        with patch("builtins.input", side_effect=input_sequence):
            with pytest.raises(ValueError):
                manual_input_dataset()

    def test_manual_input_column_count_mismatch_raises_error(self):
        """Test that rows with wrong number of columns raise ValueError."""
        input_sequence = [
            "Feature1;Target",  # Column names
            "1.0;10.0",        # Row 1
            "3.0;20.0",        # Row 2  
            "5.0;30.0",        # Row 3
            "7.0;40.0",        # Row 4
            "1.0;2.0;3.0",     # Row 5 with 3 values (should be 2, causes error)
            "",                # After error, function asks for same row again
        ]
        
        # Just test that it raises a specific error
        with patch("builtins.input", side_effect=input_sequence):
            with pytest.raises(ValueError):
                manual_input_dataset()


class TestSelectFeaturesAndTarget:
    """Test cases for select_features_and_target function."""
    
    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing selection."""
        data = np.array([
            [1.0, 2.0, 3.0, 10.0],
            [4.0, 5.0, 6.0, 20.0],
            [7.0, 8.0, 9.0, 30.0],
            [2.0, 3.0, 4.0, 15.0],
            [5.0, 6.0, 7.0, 25.0]
        ])
        columns = ["Age", "Income", "Experience", "Salary"]
        return Dataset(data=data, columns=columns)
    
    def test_automatic_selection_all_features_except_target(self, sample_dataset):
        """Test automatic selection of all features except target."""
        # Mock user preferring automatic feature selection
        # Indices: Age=0, Income=1, Experience=2, Salary=3
        # So selecting Salary (3) as target should leave features [0,1,2]
        with patch("builtins.input", return_value=""):  # Mock input for feature list
            with patch("myclt.ML.base.base_data.ask_int", return_value=4):  # Target index (4 for Salary since ask_int is 1-based)
                with patch("myclt.ML.base.base_data.ask_yes_no", return_value=False):  # Auto features
                    prepared = select_features_and_target(sample_dataset)
        
        # Verify selection
        assert isinstance(prepared, Prepareddata)
        assert prepared.target_name == "Salary"
        assert prepared.feature_names == ["Age", "Income", "Experience"]
        assert prepared.X.shape == (5, 3)  # All features except target
        assert prepared.Y.shape == (5,)
        np.testing.assert_array_equal(prepared.Y, sample_dataset.data[:, 3])

    def test_manual_feature_selection(self, sample_dataset):
        """Test manual selection of specific features."""
        # Mock user input for manual feature selection
        # Indices: Age=0 (col 1), Income=1 (col 2), Experience=2 (col 3), Salary=3 (col 4)
        # User will select columns 2 and 3 (which are Income and Experience)
        
        # Mock target selection first (selecting Salary - column 4 since ask_int is 1-based)
        with patch("myclt.ML.base.base_data.ask_int", return_value=4):
            # Mock "yes" for manual feature selection
            with patch("myclt.ML.base.base_data.ask_yes_no", return_value=True):
                # Mock feature selection input ("2,3" for Income and Experience)
                with patch("builtins.input", return_value="2,3"):
                    prepared = select_features_and_target(sample_dataset)
        
        # Verify manual selection
        assert prepared.target_name == "Salary"
        assert prepared.feature_names == ["Income", "Experience"]  # Manual selection
        assert prepared.X.shape == (5, 2)
        assert prepared.Y.shape == (5,)
        np.testing.assert_array_equal(prepared.Y, sample_dataset.data[:, 3])

    def test_selecting_target_as_feature_raises_error(self, sample_dataset):
        """Test that selecting target column as feature raises ValueError."""
        input_sequence = [
            "3",       # Select Salary as target (column 3)
            "y",       # Yes to manual feature selection
            "3,1",     # Try to include target (3) as feature
        ]
        
        with patch("builtins.input", side_effect=input_sequence):
            with patch("myclt.ML.base.base_data.ask_int", return_value=3):
                with patch("myclt.ML.base.base_data.ask_yes_no", return_value=True):
                    with pytest.raises(ValueError, match="Target column cannot be included"):
                        select_features_and_target(sample_dataset)

    def test_no_features_selected_raises_error(self, sample_dataset):
        """Test that selecting no features raises ValueError."""
        input_sequence = [
            "3",       # Select Salary as target (column 3)
            "y",       # Yes to manual feature selection
            "",        # Empty feature list
        ]
        
        with patch("builtins.input", side_effect=input_sequence):
            with patch("myclt.ML.base.base_data.ask_int", return_value=3):
                with patch("myclt.ML.base.base_data.ask_yes_no", return_value=True):
                    with pytest.raises(ValueError, match="at least 1 feature"):
                        select_features_and_target(sample_dataset)

    def test_non_numeric_feature_column_raises_error(self):
        """Test that non-numeric feature columns raise ValueError."""
        # Create dataset with one non-numeric column
        # Need numeric data - create a 2D array with mixed types by using object dtype
        data = np.array([
            [1.0, "A", 3.0, 10.0],
            [4.0, "B", 6.0, 20.0],
            [7.0, "C", 9.0, 30.0]
        ], dtype=object)
        columns = ["Age", "Category", "Experience", "Salary"]
        dataset = Dataset(data=data, columns=columns)
        
        # Mock user inputs - need to patch all input calls
        with patch("builtins.input", return_value=""):  # Mock empty input for feature list since we're using auto
            with patch("myclt.ML.base.base_data.ask_int", return_value=3):  # Target (Salary)
                with patch("myclt.ML.base.base_data.ask_yes_no", return_value=False):  # Auto features
                    with pytest.raises(ValueError, match="non-numeric values"):
                        select_features_and_target(dataset)


class TestDatasetDataClasses:
    """Test cases for Dataset and Prepareddata dataclasses."""
    
    def test_dataset_creation_and_properties(self):
        """Test basic Dataset creation and properties."""
        data = np.random.randn(10, 3)
        columns = ["Feature1", "Feature2", "Feature3"]
        
        dataset = Dataset(data=data, columns=columns)
        
        assert dataset.data.shape == (10, 3)
        assert dataset.columns == columns
        assert np.array_equal(dataset.data, data)
        
    def test_prepareddata_creation_and_properties(self):
        """Test basic Prepareddata creation and properties."""
        X = np.random.randn(10, 2)
        Y = np.random.randn(10)
        feature_names = ["Age", "Income"]
        target_name = "Salary"
        
        prepared = Prepareddata(X=X, Y=Y, feature_names=feature_names, target_name=target_name)
        
        assert prepared.X.shape == (10, 2)
        assert prepared.Y.shape == (10,)
        assert prepared.feature_names == feature_names
        assert prepared.target_name == target_name
        assert np.array_equal(prepared.X, X)
        assert np.array_equal(prepared.Y, Y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])