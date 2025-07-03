import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from src.data_loader import DataLoader

# Get the absolute path of the current script's directory (the source folder)
source_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(source_folder, "..", "data")

# Test if the folder path is acceptable:


def test_invalid_folder_path():
    invalid_path = "/invalid/nonexistent/folder/path"
    with pytest.raises(FileNotFoundError):
        DataLoader(invalid_path)


def test_file_instead_folder():
    file_name = "first_task_made.xlsx"
    invalid_path = os.path.join(folder_path, file_name)
    with pytest.raises(FileNotFoundError):
        DataLoader(invalid_path)


def test_empty_input_folder_returns_empty():
    empty_folder_path = os.path.join(folder_path, "temp_empty_folder")
    os.makedirs(empty_folder_path, exist_ok=True)

    try:
        data_loader = DataLoader(empty_folder_path)
        df_dict, feature_map = data_loader.get_dataframe()

        assert df_dict == {}, "Expected empty dictionary for df_dict"
        assert feature_map == {}, "Expected empty dictionary for feature_map"
    finally:
        if not os.listdir(empty_folder_path):  # ensure it's empty
            os.rmdir(empty_folder_path)


def test_initialization():
    data_loader = DataLoader(folder_path)
    assert data_loader.input_folder == Path(folder_path)


def test_get_dataframe():
    data_loader = DataLoader(folder_path)
    df_dict, feature_map = data_loader.get_dataframe()
    assert isinstance(df_dict, dict)
    assert isinstance(feature_map, dict)


def test_output_length():
    # Assuming you have a mock or real folder path
    data_loader = DataLoader(folder_path)  # Replace with the correct path

    # Call the get_dataframe method
    df_dict, feature_map = data_loader.get_dataframe()

    # Test if the length of the first dictionary (df_dict) is 5
    assert (
        len(df_dict) == 5
    ), f"Expected df_dict to have length 5, but got {len(df_dict)}"


def test_handle_demographic():
    clinical_folder = os.path.join(folder_path, "clinical")
    clinical_file_path = os.path.join(clinical_folder, "demographic_and_clinical.xlsx")
    data_loader = DataLoader(folder_path)
    result_name, result_df = data_loader.handle_demographic(clinical_file_path)

    assert result_name == "clinical"

    assert not result_df.empty
    assert "ID" in result_df.columns


def test_handle_cog():
    # Simulate the contents of the 'first_task_made' DataFrame
    first_task_data = {
        # Replace with appropriate IDs
        "ID": ["rn23001", "rn23004", "rn23010"],
        "CONDITION": ["stop_it", "tap_it", "tap_it"],  # Example conditions
    }
    first_task_df = pd.DataFrame(first_task_data)

    data_loader = DataLoader(folder_path)
    data_loader.first_task_made = first_task_df
    cog_folder = os.path.join(folder_path, "tasks")
    cog_file_path = os.path.join(cog_folder, "stop_it_with_code_book.xlsx")
    result_name, result_df = data_loader.handle_cog(cog_file_path)

    assert result_name == "stop_it"

    assert not result_df.empty
    assert "ID" in result_df.columns


def test_handle_physio():
    physio_folder = os.path.join(folder_path, "physio")
    physio_file_path = os.path.join(physio_folder, "statistics_hr.xlsx")
    data_loader = DataLoader(folder_path)
    result_name, result_df = data_loader.handle_physio(physio_file_path)

    assert result_name == "HR"

    assert not result_df.empty
    assert "ID" in result_df.columns


def test_filter_participants_in_physio():
    data_loader = DataLoader(folder_path)
    df = pd.DataFrame({"ID": [1, 2, 2, 3, 4], "value": [10, 20, 20, 30, 40]})
    filtered_df = data_loader.filter_participants_in_physio(
        df, 2, "removed_physio_test.txt"
    )
    assert filtered_df.shape[0] == 2


def test_gets_measurement_name_without_prefix():
    file_name = "/path/to/HR.xlsx"
    result = DataLoader.gets_measurement_name(file_name)
    assert result == "HR"


def test_invalid_feature_types_replaced_with_linear():
    csv_path = os.path.join(folder_path, "test_feature_map.csv")

    # Create a CSV with some valid and some invalid feature types
    test_data = [
        ["feature1", "linear"],
        ["feature2", "banana"],  # invalid
        ["feature3", "slinear"],
        ["feature4", "unknown"],  # invalid
        ["feature5", "linear"],
    ]

    # Save it to the expected location
    pd.DataFrame(test_data).to_csv(csv_path, index=False, header=False)

    # Run the DataLoader
    data_loader = DataLoader(folder_path)
    feature_map = data_loader.set_up_feature_map(csv_path)

    # Clean up after test
    os.remove(csv_path)

    # Assert that invalid values were replaced with 'linear'
    assert feature_map["feature1"] == "linear"
    assert feature_map["feature2"] == "linear"
    assert feature_map["feature3"] == "slinear"
    assert feature_map["feature4"] == "linear"
    assert feature_map["feature5"] == "linear"


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
