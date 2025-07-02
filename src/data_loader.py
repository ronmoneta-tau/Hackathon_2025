import pandas as pd
from pathlib import Path
import os

# /Users/user/Downloads/hackathon_2025

class DataLoader:
    """
    Base class that gets a path to a folder for use. 
    
    This class generates dictionaries of the relevant files per folder (categorizes them based
    on folder location).

    """
    def __init__(self, input_folder: str):
        """
        Initialize a new instance. 

        Parameters:
        ------
        input_folder : Path
            Input data path 
        """
        self.input_folder = Path(input_folder)
        self.clinic_data = {}
        self.task_data = {}
        self.physio_data = {}

    def get_dataframe(self):
        """
        This function gets a folder path, loads the file and than handles each file
        according to its relevant folder. 
        Eventually, it returns a dictionary of the name and the data frame. 

        Returns
        --------
        df_dict : Dict
            A dictionary of the name and its DataFrame
        Open path (loads files)-> handle cog, handle phisio -> return 4 data frames"""

        df_dict = {}
        # Walk through the folder
         # Mapping folder names to their respective handler functions
        folder_handlers = {
            'clinical': self.handle_demographic,
            'tasks': self.handle_cog,
            'physio': self.handle_physio  # Assuming physio uses same handler as tasks
        }

        for entry in os.listdir(self.input_folder):
            full_path = os.path.join(self.input_folder, entry)

            if os.path.isdir(full_path) and entry in folder_handlers:
                handler_func = folder_handlers[entry]
                for file in os.listdir(full_path):
                    file_path = os.path.join(full_path, file)
                    name, df = handler_func(file_path)
                    df_dict[name] = df

        return df_dict
    
    def handle_demographic(self, file_name):
        """
        The function opens the file, organizes the column names
        and returns the name and data frame for later use. 

        Parameter
        ---------
        file_name : str
            The name of the file that is being processed.

        Returns
        -------
        name : str
            name of the data frame
        df : DataFrame
            data frame
        """
        #"open excle, handgle merged columns, return name and dataframe"
        df = pd.read_excel(file_name, header=[0, 1])
        # df.columns = ['_'.join([str(i) for i in col if str(i) != 'nan']).strip() for col in df.columns] #changes the MultiIndex into a single-level index
        df = self.flatten_df(df)
        return "clinical", df

    def flatten_df(self, df):
        df.columns = [
            col[0].replace(' ', '-') if ('Unnamed' in str(col[1]) or '=' in str(col[1]))
            else f'{col[0]}_{col[1]}'.replace(' ', '-')
            for col in df.columns
        ]
        return df


    def handle_cog(self, file_name):
        "Gets a file name, opens as xslx file - extracts first sheet. return name and dataframe "
        excel_file = pd.ExcelFile(file_name)
        sheet_names = excel_file.sheet_names 
        df = pd.read_excel(file_name, sheet_name=sheet_names[0])
        name = self.gets_task_name(file_name)
        return name, df
    

    def gets_task_name(self, file_name):
        "from file called 'stop_it_with_code_book, extract 'stop_it'"
    def extarct_sheet(self, file_name, num=0):
        "gets the file and extracts only the sheet the user defines"
    
    def handle_physio(self, file_name):
        "Gets a file, opens as xslx - extarct 4 sheets and mergers them as one table and adda a column."
        "returns name and dataframe"
        return "x", None

    def gets_measurement_name(self, file_name):
        "from file called 'statistics_hr', extract 'hr'"
    def add_col(self, data, column_name, column_value):
        "add new column to dataframe with same value"

   
#maybe
# df.columns = pd.MultiIndex.from_tuples([
#     (col[0], '' if 'Unnamed:' in col[1] else col[1])
#     for col in df.columns
# ])
