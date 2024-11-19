from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, hour, month, year, col, when, coalesce, lit, substring, length, floor, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Start a spark session for the cluster
spark = SparkSession.builder.appName("FlightAnalysis").getOrCreate()

# Load the 2019 and 2023 datasets from local file system
df_2019 = spark.read.option("header", "true").csv("~/BigData-FlightAnalysis/Data/flights_sample_3m.csv")
df_2023 = spark.read.option("header", "true").csv("~/BigData-FlightAnalysis/Data/2023.csv")
df_2019.show(5)
df_2023.show(5)

# Define some functions to clean the data for 2019 and 2023 datasets
def clean_2019(df):
    df = df.fillna({ # Fill missing values with default values null will be set to 0
        "CANCELLED": 0,
        "DIVERTED": 0,
        "ARR_DELAY": 0.0,
        "DEP_DELAY": 0.0,
        "DEP_TIME": 0.0,
        "DISTANCE": 0.0,
        "AIRLINE": "Not Listed",
        "CANCELLATION_CODE": "None"
    })
    
    # Cast columns to appropriate types
    df = df.withColumn("Cancelled", col("CANCELLED").cast("int").cast("boolean")) \
           .withColumn("Diverted", col("DIVERTED").cast("int").cast("boolean")) \
           .withColumn("ArrDelay", col("ARR_DELAY").cast("double")) \
           .withColumn("DepDelay", col("DEP_DELAY").cast("double")) \
           .withColumn("DepTime", col("DEP_TIME").cast("double")) \
           .withColumn("Distance", col("DISTANCE").cast("double")) \
           .withColumn("Airline", col("AIRLINE").cast("string")) \
           .withColumn("DelayReason", col("CANCELLATION_CODE").cast("string"))
    return df

# Clean the 2019 dataset and print the schema and first 10 rows
cleaned_19 = clean_2019(df_2019)
'''print("Schema of '19 Data:")
cleaned_19.printSchema()
print("First 10 rows of '19 Data:")
cleaned_19.show(10)'''

# Define a function to clean the 2023 dataset
def clean_2023(df):
    df = df.fillna({ # Fill missing values with default values null will be set to 0
        "DelayMinutes": 0.0,
        "Cancelled": False,
        "Diverted": False,
        "Distance": 0.0,
        "Airline": "Not Listed",
        "DelayReason": "None"
    })

    # Cast columns to appropriate types
    df = df.withColumn("Cancelled", col("Cancelled").cast("boolean")) \
         .withColumn("Diverted", col("Diverted").cast("boolean")) \
         .withColumn("DelayMinutes", col("DelayMinutes").cast("double")) \
         .withColumn("Distance", col("Distance").cast("double")) \
         .withColumn("Airline", col("Airline").cast("string")) \
         .withColumn("DelayReason", col("DelayReason").cast("string"))
    return df

# Clean the 2023 dataset and print the schema and first 10 rows
cleaned_23 = clean_2023(df_2023)
'''print("Schema of '23 Data:")
cleaned_23.printSchema()
print("First 10 rows of '23 Data:")
cleaned_23.show(10)'''

# Prep data for clustering using basic feature engineering
def prep_features_2019(df):
  df = df.withColumn("DayofWeek", dayofweek(col("FL_DATE"))) \
        .withColumn("DepHour", floor(col("DEP_TIME") / 100).cast("int")) \
        .withColumn("Month", month(col("FL_DATE")))
  return df

prepped_19 = prep_features_2019(cleaned_19)

def prep_features_2023(df):
  return df.withColumn("DayofWeek", dayofweek(col("ScheduledDeparture"))) \
           .withColumn("DepHour", hour(col("ScheduledDeparture")))  \
           .withColumn("Month", month(col("ScheduledDeparture")))

prepped_23 = prep_features_2023(cleaned_23)

# Add a Binary Classification Label: IsDelayed to Datasets
delay_threshold = 5 # Minutes before a flight is considered delayed
prepped_19 = prepped_19.withColumn("IsDelayed", when(col("ArrDelay") > delay_threshold, 1).otherwise(0))
prepped_23 = prepped_23.withColumn("IsDelayed", when(col("DelayMinutes") > delay_threshold, 1).otherwise(0))

# Modify VectorAssembler to handle potential issues with 'Distance' and replace inf values with 0.0
cluster_assembler_19 = VectorAssembler(inputCols=["DayofWeek", "DepHour", "Month", "Distance"], outputCol="features", handleInvalid="keep")
clustered_19 = cluster_assembler_19.transform(prepped_19).select("features", "DelayReason").replace(float('inf'), 0.0, subset=['features']).replace(float('NaN'), 0.0, subset=['features'])

cluster_assembler_23 = VectorAssembler(inputCols=["DayofWeek", "DepHour", "Month", "Distance"], outputCol="features", handleInvalid="keep")
clustered_23 = cluster_assembler_23.transform(prepped_23).select("features", "DelayReason").replace(float('inf'), 0.0, subset=['features']).replace(float('nan'), 0.0, subset=['features'])

# Train KMeans models
kmeans = KMeans(k=5, seed=42)
kmeans_model_19 = kmeans.fit(clustered_19)
kmeans_model_23 = kmeans.fit(clustered_23)

# Make predictions based on the model
kmeans_predictions_19 = kmeans_model_19.transform(clustered_19)
kmeans_predictions_23 = kmeans_model_23.transform(clustered_23)

# Evaluate Clustering
cluster_eval = ClusteringEvaluator(predictionCol="prediction", featuresCol="features", metricName="silhouette") # Changed predictionCol to "prediction"
silhouette_score_19 = cluster_eval.evaluate(kmeans_predictions_19)
silhouette_score_23 = cluster_eval.evaluate(kmeans_predictions_23)

print(f"Silhouette Score (2019): {silhouette_score_19}")
print(f"Silhouette Score (2023): {silhouette_score_23}")

# Also change predictionCol to "prediction" in the groupBy for top reasons
top_reasons_19 = kmeans_predictions_19.groupBy("prediction", "DelayReason").count().orderBy("prediction", "count", ascending=False) # Changed "cluster" to "prediction"
top_reasons_23 = kmeans_predictions_23.groupBy("prediction", "DelayReason").count().orderBy("prediction", "count", ascending=False) # Changed "cluster" to "prediction"

print("Top Reasons for Delays (2019):")
top_reasons_19.show(truncate=False)

print("Top Reasons for Delays (2023):")
top_reasons_23.show(truncate=False)

# For simplicity, we will use the 2023 data to predict cancellations in 2024
future_data = spark.createDataFrame([
    (1, 10, 6, 500, "Delta"), 
    (5, 15, 12, 1000, "United"),
    (2, 7, 9, 750, "Southwest Airlines"),
    (7, 20, 2, 4000, "American Airlines"),
    (5, 8, 4, 5599, "Air France")], 
     ["DayofWeek", "DepHour", "Month", "Distance", "Airline"])

future_data = cluster_assembler_23.transform(future_data)
future_predictions = kmeans_model_23.transform(future_data)
future_predictions.show()

# Function to convert a vector to a string representation
def vector_to_string(vector):
  if vector is not None:
    return str(vector.toArray().tolist())  # Convert to a list and then to a string
  else:
    return None

# Register the UDF
vector_to_string_udf = udf(vector_to_string, StringType())

# Convert 'features' column to string before saving
kmeans_predictions_19 = kmeans_predictions_19.withColumn("features_str", vector_to_string_udf("features")).drop("features")
kmeans_predictions_23 = kmeans_predictions_23.withColumn("features_str", vector_to_string_udf("features")).drop("features")
future_predictions = future_predictions.withColumn("features_str", vector_to_string_udf("features")).drop("features")

# Save cluster results to a CSV file
kmeans_predictions_19.write.format("csv").option("header", "true").save("~/BigData-FlightAnalysis/Data/kmeans_predictions_19.csv")
kmeans_predictions_23.write.format("csv").option("header", "true").save("~/BigData-FlightAnalysis/Data/kmeans_predictions_23.csv")

# Save future predictions
future_predictions.write.format("csv").option("header", "true").save("~/BigData-FlightAnalysis/Data/future_predictions.csv")

# Stop the spark session
spark.stop()