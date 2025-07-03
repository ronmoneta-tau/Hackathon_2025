import os
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader

folder_path = "/Users/user/Downloads/hackathon_2025"

# Test if the folder path is acceptable:


def test_invalid_folder_path():
    invalid_path = "/invalid/nonexistent/folder/path"
    with pytest.raises(FileNotFoundError):
        DataLoader(invalid_path)


def test_file_instead_folder():
    invalid_path = (
        "/Users/user/Documents/2nd_Degree/Courses/Python/parse_subject_names.ipynb"
    )
    with pytest.raises(FileNotFoundError):
        DataLoader(invalid_path)


def test_empty_input_folder_returns_empty():
    empty_folder_path = "/Users/user/Documents/2nd_Degree/Courses/Python/empty_folder"
    data_loader = DataLoader(empty_folder_path)

    df_dict, feature_map = data_loader.get_dataframe()

    assert df_dict == {}, "Expected empty dictionary for df_dict"
    assert feature_map == {}, "Expected empty dictionary for feature_map"


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
    clinical_file_path = os.path.join(
        "/Users/user/Downloads/hackathon_2025/clinical",
        "demographic_and_clinical.xlsx")
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

    cog_file_path = os.path.join(
        "/Users/user/Downloads/hackathon_2025/tasks",
        "stop_it_with_code_book.xlsx")
    result_name, result_df = data_loader.handle_cog(cog_file_path)

    assert result_name == "stop_it"

    assert not result_df.empty
    assert "ID" in result_df.columns


def test_handle_physio():
    clinical_file_path = os.path.join(
        "/Users/user/Downloads/hackathon_2025/physio", "statistics_hr.xlsx"
    )
    data_loader = DataLoader(folder_path)
    result_name, result_df = data_loader.handle_physio(clinical_file_path)

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


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
