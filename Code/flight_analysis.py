from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, hour, month, col, when, coalesce, lit, floor, udf, desc
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Start a spark session for the cluster
spark = SparkSession.builder.appName("FlightAnalysis").getOrCreate()

df_2019 = spark.read.option("header", "true").csv("~/BigData-FlightAnalysis/Data/2019.csv")
df_2023 = spark.read.option("header", "true").csv("~/BigData-FlightAnalysis/Data/2023.csv")

# Convert to Pandas DataFrame for better display
df_2019_pd = df_2019.limit(5).toPandas()
df_2023_pd = df_2023.limit(5).toPandas()

# Display the DataFrames
print("2019 Data:")
print(df_2019_pd)
print("2023 Data:")
print(df_2023_pd)

# Map cancellation codes to reasons
cancellation_reasons = {
    "A": "Carrier Caused",
    "B": "Weather",
    "C": "National Aviation System",
    "D": "Security",
    "None": "No Cancellation"
}

# Convert the dictionary to a case-when expression
cancellation_expr = "CASE " + " ".join(
    f"WHEN CANCELLATION_CODE = '{code}' THEN '{reason}'"
    for code, reason in cancellation_reasons.items()
) + " END AS CancellationReason"

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
           .withColumn("DelayReason", col("CANCELLATION_CODE").cast("string")) \
           .withColumn("CancellationReason", expr(cancellation_expr))  # Map cancellation codes

    df = df.select("FL_DATE", "DepTime", "ArrDelay", "Cancelled", "Diverted", "Distance", "Airline", "DelayReason", "CancellationReason")
    return df

cleaned_19 = clean_2019(df_2019)
cleaned_19_pd = cleaned_19.limit(10).toPandas()

def clean_2023(df):
  df = df.fillna({
        "DelayMinutes": 0.0,
        "Cancelled": False,
        "Diverted": False,
        "Distance": 0.0,
        "Airline": "Not Listed",
        "DelayReason": "None"
  })

  df = df.withColumn("Cancelled", col("Cancelled").cast("boolean")) \
         .withColumn("Diverted", col("Diverted").cast("boolean")) \
         .withColumn("DelayMinutes", col("DelayMinutes").cast("double")) \
         .withColumn("Distance", col("Distance").cast("double")) \
         .withColumn("Airline", col("Airline").cast("string")) \
         .withColumn("DelayReason", col("DelayReason").cast("string"))

  df = df.select("ScheduledDeparture", "DelayMinutes", "Cancelled", "Diverted", "Distance", "Airline", "DelayReason")
  return df

cleaned_23 = clean_2023(df_2023)
cleaned_23_pd = cleaned_23.limit(10).toPandas()

print("First 10 rows of '19 Data:")
print(cleaned_19_pd)
print("First 10 rows of '19 Data:")
print(cleaned_23_pd)

def prep_features_2019(df):
  df = df.withColumn("DayofWeek", dayofweek(col("FL_DATE"))) \
        .withColumn("DepHour", floor(col("DepTime") / 100).cast("int")) \
        .withColumn("Month", month(col("FL_DATE")))
  return df

prepped_19 = prep_features_2019(cleaned_19)
prepped_19_pd = prepped_19.limit(10).toPandas()

def prep_features_2023(df):
  return df.withColumn("DayofWeek", dayofweek(col("ScheduledDeparture"))) \
           .withColumn("DepHour", hour(col("ScheduledDeparture")))  \
           .withColumn("Month", month(col("ScheduledDeparture")))

prepped_23 = prep_features_2023(cleaned_23)
prepped_23_pd = prepped_23.limit(10).toPandas()

print("First 10 rows of '23 Data After Feature Engineering:")
print(prepped_23_pd)
print("First 10 rows of '19 Data After Feature Engineering:")
print(prepped_19_pd)

# Add a Binary Classification Label: IsDelayed to Datasets
delay_threshold = 5 # Minutes before a flight is considered delayed
prepped_19 = prepped_19.withColumn("IsDelayed", when(col("ArrDelay") > delay_threshold, 1).otherwise(0))
prepped_23 = prepped_23.withColumn("IsDelayed", when(col("DelayMinutes") > delay_threshold, 1).otherwise(0))

# Modify VectorAssembler to handle potential issues with 'Distance' and replace inf values with 0.0
cluster_assembler_19 = VectorAssembler(inputCols=["DayofWeek", "DepHour", "Month", "Distance"], outputCol="features", handleInvalid="keep")
clustered_19 = cluster_assembler_19.transform(prepped_19).select("features", "CancellationReason").replace(float('inf'), 0.0, subset=['features']).replace(float('NaN'), 0.0, subset=['features'])

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
cluster_eval = ClusteringEvaluator(predictionCol="prediction", featuresCol="features", metricName="silhouette")
silhouette_score_19 = cluster_eval.evaluate(kmeans_predictions_19)
silhouette_score_23 = cluster_eval.evaluate(kmeans_predictions_23)
top_reasons_19 = kmeans_predictions_19.groupBy("prediction", "CancellationReason").count().orderBy("prediction", "count", ascending=False)
top_reasons_23 = kmeans_predictions_23.groupBy("prediction", "DelayReason").count().orderBy("prediction", "count", ascending=False)

# Convert data to pandas
top_reasons_19_pd = top_reasons_19.limit(20).toPandas()
top_reasons_23_pd = top_reasons_23.limit(20).toPandas()

# Display clustering results and evaluations
# Silhouette scores bar plot
years = ['2019', '2023']
scores = [silhouette_score_19, silhouette_score_23]

plt.bar(years, scores, color=['blue', 'orange'])
plt.title('Silhouette Scores for KMeans Clustering')
plt.ylabel('Silhouette Score')
plt.xlabel('Year')
plt.ylim(0, 1)  # Silhouette scores range between -1 and 1
plt.show()

# Define a sample dataset for predictions
future_data = spark.createDataFrame([
    (1, 10, 6, 500, "Delta"),
    (5, 15, 12, 1000, "United"),
    (2, 7, 9, 750, "Southwest Airlines"),
    (7, 20, 2, 4000, "American Airlines"),
    (5, 8, 4, 175, "Frontier Airlines")],
     ["DayofWeek", "DepHour", "Month", "Distance", "Airline"])

# Aggregate cancellation counts by airline for 2019 and 2023
cancelled_19 = prepped_19.filter(col("Cancelled") == 1) \
    .groupBy("Airline").count() \
    .orderBy(desc("count"))

cancelled_23 = prepped_23.filter(col("Cancelled") == 1) \
    .groupBy("Airline").count() \
    .orderBy(desc("count"))

# Get TopK airlines with the most cancellations
K = 5
top_k_airlines_19 = cancelled_19.limit(K)
top_k_airlines_23 = cancelled_23.limit(K)
top_k_airlines_19_pd = top_k_airlines_19.toPandas()
top_k_airlines_23_pd = top_k_airlines_23.toPandas()

# Show results
print("Top K Airlines by Cancellations (2019):")
print(top_k_airlines_19_pd)
print("Top K Airlines by Cancellations (2023):")
print(top_k_airlines_23_pd)

# Filter data for TopK airlines
top_airlines = [row["Airline"] for row in top_k_airlines_23.collect()]
future_data_filtered = future_data.filter(col("Airline").isin(top_airlines))

#Ensure the features column is created for the filtered data
future_data_filtered = cluster_assembler_23.transform(future_data_filtered)
future_predictions_filtered = kmeans_model_23.transform(future_data_filtered)
future_predictions_filtered_pd = future_predictions_filtered.toPandas()

# Display prediction results
# Top reasons for delays (2019)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_reasons_19_pd, x='prediction', y='count', hue='CancellationReason', palette='viridis')
plt.title('Top Reasons for Delays/Cancellations (2019)')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Top reasons for delays (2023)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_reasons_23_pd, x='prediction', y='count', hue='DelayReason', palette='viridis')
plt.title('Top Reasons for Delays (2023)')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Function to convert a vector to a string representation
def vector_to_string(vector):
    if vector is not None:
        return str(vector.toArray().tolist())  # Convert to a list and then to a string
    else:
        return None

# Register the UDF
vector_to_string_udf = udf(vector_to_string, StringType())

kmeans_predictions_19_with_str = kmeans_predictions_19.withColumn("features_str", vector_to_string_udf("features")).drop("features")
kmeans_predictions_23_with_str = kmeans_predictions_23.withColumn("features_str", vector_to_string_udf("features")).drop("features")

# For future predictions
future_predictions_with_str = future_predictions_filtered.withColumn("features_str", vector_to_string_udf("features")).drop("features")

# Save all results to Google Drive
kmeans_predictions_19_with_str.write.format("csv").option("header", "true").mode("overwrite").save("/content/drive/MyDrive/Data/kmeans_predictions_19.csv")
kmeans_predictions_23_with_str.write.format("csv").option("header", "true").mode("overwrite").save("/content/drive/MyDrive/Data/kmeans_predictions_23.csv")
future_predictions_with_str.write.format("csv").option("header", "true").mode("overwrite").save("/content/drive/MyDrive/Data/future_predictions.csv")