
import pandas as pd
import DataLoader


class PhysioDataManager:

    # Maps file names to their corresponding DataFrames (from DataLoader).
    physio_files = {}

    def __init__(self, physio_files):
        self.physio_files = physio_files

    def get_file(self, file_name: str) -> pd.DataFrame:
        # Returns the physio DataFrame by file name.
        if file_name in self.physio_files:
            return self.physio_files[file_name]

    def list_available_signals(self, file_name: str) -> list:
        # Returns column names for a given physio file.
        if file_name in self.physio_files:
            return self.physio_files[file_name].columns.tolist()

    # def filter_by_stage(self, stage_name: str):
    #     # Returns subset of data matching a certain stage( if labeled in files).
    #     pass

