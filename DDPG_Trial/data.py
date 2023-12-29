import pandas as pd

class Import_Data(object):
    def sin_data(self):
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

    def chirp_data(self):
        csv_file_path_1 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Chirp/Chirp_Freq_01-06_2min.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

        csv_file_path_2 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Chirp/Chirp_Freq_01-06_2min_Iter02.CSV" # Replace with the actual path to your Excel file
        df_2 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column
    
    def random_step_data(self):
        csv_file_path_1 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Random Step/Random_Step_6min.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

    def sinusoid_data_2(self):
        csv_file_path_1 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Sinusoid/12-9/freq_0.1_7min.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,7,8,9], nrows=20000) # Load the Excel sheet, excluding the specified column

        csv_file_path_2 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Sinusoid/12-9/freq_0.2_7min.CSV" # Replace with the actual path to your Excel file
        df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[5,7,8,9], nrows=20000) # Load the Excel sheet, excluding the specified column

        csv_file_path_3 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Sinusoid/12-9/freq_0.05_7min.CSV" # Replace with the actual path to your Excel file
        df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[5,7,8,9], nrows=20000) # Load the Excel sheet, excluding the specified column

    def random_step_data_2(self):
        csv_file_path_1 = "C:/Users/keegh/Documents/Orbtronics_Agri_Sensor/DRL_Test_Algorithms/training data/Random Step/random_ramp_test_17min.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[3,4,6,7], nrows=60000) # Load the Excel sheet, excluding the specified column

###########################################GAZI MODS###############################################################################################
    def sinusoid_data_3(self):
        csv_file_path_1 = r"C:\Users\sakib\Documents\DRL_Test_Algorithms\training data\12_23_data\Sinusoid\Freq_.05_15min.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, nrows=60000) # Load the Excel sheet, excluding the specified column
        return df_1

    def random_step_data_3(self):
        csv_file_path_1 = r"C:\Users\sakib\Documents\DRL_Test_Algorithms\training data\12_23_data\Ramp_test_data_85\ramp_test_16min.CSV" # Replace with the actual path to your Excel file
        df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, nrows=60000) # Load the Excel sheet, excluding the specified column
        return df_1