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
        self.first_task_made = None
        self.feature_map = None
        self.clinic_data = {}
        self.task_data = {}
        self.physio_data = {}

# need to load the order of tasks and add it to the cog pd. 
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

        # find sxcel file and save it
        for item in os.listdir(self.input_folder):
            item_path = os.path.join(self.input_folder, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.xls', '.xlsx')):
                self.first_task_made = pd.read_excel(item_path)
            elif os.path.isfile(item_path) and item.lower().endswith(('.csv')):
                feature_df = pd.read_csv(item_path, header=None)  # Read CSV without header
                self.feature_map = pd.Series(feature_df[1].values, index=feature_df[0].values).to_dict()
                # check if value isn't a real method, if not chose default

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
    @staticmethod  
    def flatten_df(df):
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
        #change the name of the first column:
        df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
        name = self.gets_task_name(file_name)

        # add column of first and second
        df = self.map_condition_column(df, name)
        
        return name, df
    
    def map_condition_column(self, df_target, target):
        condition_col = 'CONDITION'
        participant_col = 'ID'  # Update this if your ID column has a different name

        # Step 1: Get mapping per participant from first_task_made
        cond_map = self.first_task_made[[participant_col, condition_col]].drop_duplicates()

        # Step 2: Map 'Tap It'/'Stop It' to 'First'/'Second'
        cond_map[condition_col] = cond_map[condition_col].map(
            lambda x: 'First' if x == target else 'Second'
        )

        # Step 3: Merge this info into df_target
        df_target = df_target.copy()
        df_target = df_target.merge(cond_map, on=participant_col, how='left')

        return df_target
    
    @staticmethod
    def gets_task_name(file_name):
        "from file called 'stop_it_with_code_book, extract 'stop_it'"
        base_name = file_name.split('/')[-1]
        return base_name.split('_with_code_book')[0]
    
    def handle_physio(self, file_name):
        "Gets a file, opens as xslx - extarct 4 sheets and mergers them as one table and adda a column."
        "returns name and dataframe"
        df_list = []
        excel_file = pd.ExcelFile(file_name)
        sheet_names = excel_file.sheet_names 
        for sheet in sheet_names:
            sheet_df = pd.read_excel(file_name, sheet_name=sheet)
            sheet_df['sheet_name'] = sheet
            df_list.append(sheet_df)
        df = pd.concat(df_list, ignore_index=True)
        df.rename(columns={df.columns[0]: 'ID'}, inplace=True)

        name = self.gets_measurement_name(file_name)

        filter_df = self.filter_participants_in_physio(df, len(sheet_names), "removed_" + name)

        return name, filter_df

    def filter_participants_in_physio(self, df, num, file_name, participant_col='ID'):
        """
        """
        counts = df[participant_col].value_counts()
        valid_participants = counts[counts == num].index
        removed_participants = counts[counts != num].index.tolist()
        filtered_df = df[df[participant_col].isin(valid_participants)].copy()
        csv_path = os.path.join(self.input_folder, file_name)
        pd.DataFrame(removed_participants, columns=[participant_col]).to_csv(csv_path, index=False)
        filtered_df = filtered_df.reset_index(drop=True)

        return filtered_df

    @staticmethod
    def gets_measurement_name(file_name):
        "from file called 'statistics_hr', extract 'hr'"
        base_name = file_name.split('/')[-1]
        base_name = base_name.replace("statistics_", "").upper()
        return base_name.split(".")[0]