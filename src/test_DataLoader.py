
folder_path = "/Users/user/Downloads/hackathon_2025"
import data_loader

if __name__ == "__main__":
    data_loader_instance = data_loader.DataLoader(folder_path)
    df_dict = data_loader_instance.get_dataframe()

