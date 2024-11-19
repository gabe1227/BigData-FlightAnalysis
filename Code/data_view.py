import pandas as pd

def data_view():
    pd.set_option('display.max_columns', None)
    # Load the datasets
    data_2023 = pd.read_csv("~/BigData-FlightAnalysis/Data/2023.csv")
    data_2019 = pd.read_csv("~/BigData-FlightAnalysis/Data/flights_sample_3m.csv")
    
    # Display the first few rows of each dataset
    print("2023 Data:")
    print(data_2023.head())
    print("\n2019 Data:")
    print(data_2019.head())

# Call the function to display the data
data_view()