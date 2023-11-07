import pandas as pd

class Import_Data(object):
    def data(self):
        csv_file_path_1 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_2 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
        df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_3 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
        df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_4 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
        df_4 = pd.read_csv(csv_file_path_4, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_5 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
        df_5 = pd.read_csv(csv_file_path_5, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_6 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
        df_6 = pd.read_csv(csv_file_path_6, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_7 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
        df_7 = pd.read_csv(csv_file_path_7, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_8 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
        df_8 = pd.read_csv(csv_file_path_8, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column

        csv_file_path_9 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
        df_9 = pd.read_csv(csv_file_path_9, skiprows=13, header=0, usecols=[5,6,7], nrows=13000) # Load the Excel sheet, excluding the specified column