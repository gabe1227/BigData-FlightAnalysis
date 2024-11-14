import pandas as pd

def data_view():
    pd.set_option('display.max_columns', None)
    # Load the datasets
    data_2023 = pd.read_csv("../Data/2023_flight_data/flight_delays.csv")
    data_2007 = pd.read_csv("../Data/2007_flight_data/2007.csv")
    
    # Display the first few rows of each dataset
    print("2023 Data:")
    print(data_2023.head())
    print("\n2007 Data:")
    print(data_2007.head())

# Call the function to display the data
data_view()