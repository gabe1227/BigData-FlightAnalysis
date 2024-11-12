from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, when
from pyspark.sql.types import BooleanType, IntegerType

from Code.parse import clean_2023_data, clean_2007_data

# Initialize Spark session
spark = SparkSession.builder.appName("FlightDelayPreprocessing").getOrCreate()

# Load the 2023 and 2007 datasets
# Replace file paths with the actual paths where your CSV files are stored
df_2023 = spark.read.option("header", "true").csv("flight_delays_2023_cleaned.csv")
df_2007 = spark.read.option("header", "true").csv("flight_delays_2007_cleaned.csv")

# Clean the 2023 and 2007 datasets individually
df_2023_clean = clean_2023_data(df_2023)
df_2007_clean = clean_2007_data(df_2007)

# Show the cleaned 2023 dataset
df_2023_clean.show()
