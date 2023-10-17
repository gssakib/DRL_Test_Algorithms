import pandas as pd

def data():
    # Load the Excel file and skip a specific column
    csv_file_path_1 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/Closed_Loop_experiments/8-18 Experiments/diameter&RPM_SP0.2.CSV"  # Replace with the actual path to your Excel file

    # Load the Excel sheet, excluding the specified column

    df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[3,4], nrows=5000)
    df_1['Setpoint'] = 0.2

    csv_file_path_2 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/Closed_Loop_experiments/8-18 Experiments/diameter&RPM_SP0.4.CSV"  # Replace with the actual path to your Excel file

    # Load the Excel sheet, excluding the specified column

    df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[3,4], nrows=5000)
    df_2['Setpoint'] = 0.4

    csv_file_path_3 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/Closed_Loop_experiments/8-18 Experiments/diameter&RPM_SP0.6.CSV"  # Replace with the actual path to your Excel file

    # Load the Excel sheet, excluding the specified column

    df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[3,4], nrows=5000)
    df_3['Setpoint'] = 0.6

    # Display the DataFrame
    print(df_1)
    print(df_2)
    print(df_3)

    df = pd.concat([df_1, df_2, df_3], axis=0)

    return df
