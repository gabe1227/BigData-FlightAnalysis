from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, BooleanType, StringType, TimestampType
from pyspark.sql.functions import col, when, to_timestamp, date_format
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("FlightDelayPreprocessing").getOrCreate()

# Load the 2023 and 2007 datasets
# Replace file paths with the actual paths where your CSV files are stored
df_2023 = spark.read.option("header", "true").csv("flight_delays.csv")
df_2007 = spark.read.option("header", "true").csv("flight_delays_2007.csv")

# Clean the 2023 and 2007 datasets individually
def clean_2023_data(df):
    # Convert dates and times to Timestamp
    df = df.withColumn("ScheduledDeparture", to_timestamp("ScheduledDeparture", "M/d/yyyy H:mm"))
    df = df.withColumn("ActualDeparture", to_timestamp("ActualDeparture", "M/d/yyyy H:mm"))
    df = df.withColumn("ScheduledArrival", to_timestamp("ScheduledArrival", "M/d/yyyy H:mm"))
    df = df.withColumn("ActualArrival", to_timestamp("ActualArrival", "M/d/yyyy H:mm"))

    # Handle DelayMinutes as Integer and clean negative values if needed
    df = df.withColumn("DelayMinutes", col("DelayMinutes").cast(IntegerType()))
    df = df.withColumn("DelayMinutes", when(col("DelayMinutes") < 0, None).otherwise(col("DelayMinutes")))

    # Fill missing values in important columns
    df = df.fillna({"DelayReason": "Unknown", "Cancelled": "FALSE", "Diverted": "FALSE"})

    # Convert boolean-like columns to actual BooleanType
    df = df.withColumn("Cancelled", col("Cancelled").cast(BooleanType()))
    df = df.withColumn("Diverted", col("Diverted").cast(BooleanType()))

    # Ensure Distance is an integer
    df = df.withColumn("Distance", col("Distance").cast(IntegerType()))
    
    return df

def clean_2007_data(df):
    # Convert dates and times, using Epoch-style minutes (e.g., 1232 is 12:32)
    df = df.withColumn("DepTime", F.expr("CAST(FLOOR(DepTime / 100) AS STRING) || ':' || CAST(MOD(DepTime, 100) AS STRING)").cast("Timestamp"))
    df = df.withColumn("ArrTime", F.expr("CAST(FLOOR(ArrTime / 100) AS STRING) || ':' || CAST(MOD(ArrTime, 100) AS STRING)").cast("Timestamp"))

    # Fill missing values in crucial columns
    df = df.fillna({"CancellationCode": "None", "Cancelled": 0, "Diverted": 0})

    # Convert Cancelled and Diverted to Boolean
    df = df.withColumn("Cancelled", col("Cancelled").cast(BooleanType()))
    df = df.withColumn("Diverted", col("Diverted").cast(BooleanType()))

    # Convert distance to Integer
    df = df.withColumn("Distance", col("Distance").cast(IntegerType()))
    
    # Handle delay columns and replace negative values if required
    delay_columns = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    for col_name in delay_columns:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))
        df = df.withColumn(col_name, when(col(col_name) < 0, None).otherwise(col(col_name)))
    
    return df

# Clean each dataset
df_2023_clean = clean_2023_data(df_2023)
df_2007_clean = clean_2007_data(df_2007)

def add_features(df, is_2023=True):
    # Extract month and day for seasonality analysis
    df = df.withColumn("Month", date_format("ScheduledDeparture" if is_2023 else "DepTime", "MM").cast(IntegerType()))
    df = df.withColumn("Day", date_format("ScheduledDeparture" if is_2023 else "DepTime", "dd").cast(IntegerType()))

    # Categorize delays
    df = df.withColumn("DelayCategory",
                       when(col("DelayMinutes" if is_2023 else "ArrDelay") < 15, "On Time")
                       .when(col("DelayMinutes" if is_2023 else "ArrDelay") < 60, "Minor Delay")
                       .when(col("DelayMinutes" if is_2023 else "ArrDelay") < 180, "Moderate Delay")
                       .otherwise("Severe Delay"))

    return df

# Add features
df_2023_final = add_features(df_2023_clean, is_2023=True)
df_2007_final = add_features(df_2007_clean, is_2023=False)

# Print the first 5 rows of each dataset
df_2023_final.show(5)
df_2007_final.show(5)

# Save the cleaned datasets to CSV files
# Replace file paths with the actual paths where you want to save the cleaned CSV files
df_2023_final.write.option("header", "true").csv("flight_delays_2023_cleaned.csv")
df_2007_final.write.option("header", "true").csv("flight_delays_2007_cleaned.csv")