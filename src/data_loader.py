import pandas as pd
from pathlib import Path
import os


class DataLoader:
    """
    Base class that gets a path to a folder for use.

    This class generates dictionaries of the relevant files per folder (categorizes them based
    on folder location).

    """

    def __init__(self, input_folder: str):
        """
        Initializes the DataLoader with a specified input folder.

        Parameters
        ----------
        input_folder : str
            The path to the root directory containing the data files.
        """
        self.input_folder = Path(input_folder)
        if not self.input_folder.exists() or not self.input_folder.is_dir():
            raise FileNotFoundError(
                f"Input folder does not exist or is not a directory: {self.input_folder}"
            )
        self.first_task_made = None
        self.feature_map = {}

    def get_dataframe(self):
        """
        Loads all relevant data files from the input folder.

        Identifies Excel and CSV files in the root directory and processes them accordingly.
        Then moves to subdirectories ('clinical', 'tasks', 'physio'), delegating to handler
        functions for each file type.

        Returns
        -------
        tuple
            A tuple containing:
            - dict: Mapping of dataset names to their corresponding pandas DataFrames.
            - dict: Feature mapping dictionary created from the CSV file (if found).
        """
        df_dict = {}
        # Walk through the folder
        # Mapping folder names to their respective handler functions
        folder_handlers = {
            "clinical": self.handle_demographic,
            "tasks": self.handle_cog,
            "physio": self.handle_physio,
        }

        # find excel file and save it:
        for item in os.listdir(self.input_folder):
            item_path = os.path.join(self.input_folder, item)
            if os.path.isfile(item_path) and item.lower().endswith((".xls", ".xlsx")):
                self.first_task_made = pd.read_excel(item_path)
            elif os.path.isfile(item_path) and item.lower().endswith((".csv")):
                feature_df = pd.read_csv(
                    item_path, header=None
                )  # Read CSV without header
                self.feature_map = pd.Series(
                    feature_df[1].values, index=feature_df[0].values
                ).to_dict()

        for entry in os.listdir(self.input_folder):
            full_path = os.path.join(self.input_folder, entry)

            if os.path.isdir(full_path) and entry in folder_handlers:
                handler_func = folder_handlers[entry]
                for file in os.listdir(full_path):
                    file_path = os.path.join(full_path, file)
                    name, df = handler_func(file_path)
                    df_dict[name] = df

        return df_dict, self.feature_map

    def handle_demographic(self, file_name):
        """
        Processes a clinical demographics Excel file.

        Opens the file, handles multi-level column headers, flattens them,
        and returns a named DataFrame for use in analysis.

        Parameters
        ----------
        file_name : str
            Full path to the clinical Excel file.

        Returns
        -------
        tuple
            A tuple of:
            - str: The name assigned to this dataset ("clinical").
            - pandas.DataFrame: The processed clinical data.
        """
        df = pd.read_excel(file_name, header=[0, 1])
        df = self.flatten_df(df)
        return "clinical", df

    @staticmethod
    def flatten_df(df):
        """
        Flattens a MultiIndex DataFrame by merging multi-level column headers into single strings.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame with MultiIndex columns.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with flattened column names.
        """
        df.columns = [
            (
                col[0].replace(" ", "-")
                if ("Unnamed" in str(col[1]) or "=" in str(col[1]))
                else f"{col[0]}_{col[1]}".replace(" ", "-")
            )
            for col in df.columns
        ]
        return df

    def handle_cog(self, file_name):
        """
        Processes a cognitive task Excel file.

        Loads the first sheet of the file, renames the participant ID column,
        extracts the task name from the filename, and maps a condition
        ('First' or 'Second') for each participant.

        Parameters
        ----------
        file_name : str
            Full path to the task Excel file.

        Returns
        -------
        tuple
            A tuple of:
            - str: Name of the cognitive task.
            - pandas.DataFrame: Processed cognitive data with condition labels.
        """
        excel_file = pd.ExcelFile(file_name)
        sheet_names = excel_file.sheet_names
        df = pd.read_excel(file_name, sheet_name=sheet_names[0])
        df.rename(columns={df.columns[0]: "ID"}, inplace=True)
        name = self.gets_task_name(file_name)
        df = self.map_condition_column(df, name)
        return name, df

    def map_condition_column(self, df_target, target):
        """
        Adds a 'CONDITION' column to a DataFrame based on the first task performed by each participant.

        Maps the current task name to 'First' or 'Second' by comparing it with previously loaded task metadata.

        Parameters
        ----------
        df_target : pandas.DataFrame
            The DataFrame to which the condition column will be added.

        target : str
            The name of the current task being processed.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with a new 'CONDITION' column.
        """
        condition_col = "CONDITION"
        participant_col = "ID"  # Update this if your ID column has a different name
        cond_map = self.first_task_made[
            [participant_col, condition_col]
        ].drop_duplicates()
        cond_map[condition_col] = cond_map[condition_col].map(
            lambda x: "First" if x == target else "Second"
        )
        df_target = df_target.copy()
        df_target = df_target.merge(cond_map, on=participant_col, how="left")

        return df_target

    @staticmethod
    def gets_task_name(file_name):
        """
        Extracts the task name from a filename.

        Assumes filenames are in the format '<task>_with_code_book.xlsx'.

        Parameters
        ----------
        file_name : str
            Full path or name of the file.

        Returns
        -------
        str
            The extracted task name.
        """
        base_name = Path(file_name).stem
        return base_name.split("_with_code_book")[0]

    def handle_physio(self, file_name):
        """
        Processes a physiological measurement Excel file.

        Reads all sheets in the file, adds a sheet identifier column,
        merges the sheets into one DataFrame, renames the ID column,
        and filters participants with missing measurements.

        Parameters
        ----------
        file_name : str
            Full path to the physiological Excel file.

        Returns
        -------
        tuple
            A tuple of:
            - str: Name of the physiological measurement.
            - pandas.DataFrame: Filtered DataFrame containing valid participant data.
        """
        df_list = []
        excel_file = pd.ExcelFile(file_name)
        sheet_names = excel_file.sheet_names
        for sheet in sheet_names:
            sheet_df = pd.read_excel(file_name, sheet_name=sheet)
            sheet_df["sheet_name"] = sheet
            df_list.append(sheet_df)
        df = pd.concat(df_list, ignore_index=True)
        df.rename(columns={df.columns[0]: "ID"}, inplace=True)

        name = self.gets_measurement_name(file_name)

        filter_df = self.filter_participants_in_physio(
            df, len(sheet_names), "removed_" + name
        )

        return name, filter_df

    def filter_participants_in_physio(self, df, num, file_name, participant_col="ID"):
        """
        Filters out participants who do not have complete data across all sheets.

        Also saves the list of removed participants to a text file.

        Parameters
        ----------
        df : pandas.DataFrame
            The combined DataFrame of all physio sheets.

        num : int
            The expected number of entries per participant (i.e., number of sheets).

        file_name : str
            Name for the file to store excluded participants.

        participant_col : str, optional
            Column name representing participant IDs (default is 'ID').

        Returns
        -------
        pandas.DataFrame
            A filtered DataFrame containing only participants with full data.
        """
        counts = df[participant_col].value_counts()
        valid_participants = counts[counts == num].index
        removed_participants = counts[counts != num].index.tolist()
        filtered_df = df[df[participant_col].isin(valid_participants)].copy()
        csv_path = Path(self.input_folder) / file_name
        pd.DataFrame(removed_participants, columns=[participant_col]).to_csv(
            csv_path, index=False
        )
        filtered_df = filtered_df.reset_index(drop=True)

        return filtered_df

    @staticmethod
    def gets_measurement_name(file_name):
        """
        Extracts the physiological measurement name from a filename.

        Assumes filenames are in the format 'statistics_<measurement>.xlsx'.

        Parameters
        ----------
        file_name : str
            Full path or name of the file.

        Returns
        -------
        str
            The extracted measurement name in uppercase.
        """
        base_name = Path(file_name).stem
        if base_name.startswith("statistics_"):
            return base_name.replace("statistics_", "").upper()
        return base_name.upper()
