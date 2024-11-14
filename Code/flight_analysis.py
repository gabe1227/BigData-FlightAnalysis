from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, hour, month, col, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator

# Initialize Spark session with HDFS configuration
hdfs_host = "hdfs://providence.cs.colostate.edu:30222"

spark = SparkSession.builder \
    .appName("FlightDelayAnalysis") \
    .config("spark.hadoop.fs.defaultFS", hdfs_host) \
    .getOrCreate()

# Load the 2023 and 2007 datasets from HDFS
df_2023 = spark.read.option("header", "true").csv(hdfs_host + "/project/input/flight_delays.csv")
df_2007 = spark.read.option("header", "true").csv(hdfs_host + "/project/input/2007.csv")

# Clean data and select relevant columns
def clean_data(df):
    return df.withColumn("DelayMinutes", col("DelayMinutes").cast("double")) \
             .withColumn("Cancelled", col("Cancelled").cast("boolean")) \
             .withColumn("Diverted", col("Diverted").cast("boolean"))

df_2023_clean = clean_data(df_2023)
df_2007_clean = clean_data(df_2007)

# Feature Engineering for Both Datasets
def prepare_features(df):
    return df.withColumn("DayOfWeek", dayofweek("ScheduledDeparture")) \
             .withColumn("DepHour", hour("ScheduledDeparture")) \
             .withColumn("Month", month("ScheduledDeparture"))

df_2023_clean = prepare_features(df_2023_clean)
df_2007_clean = prepare_features(df_2007_clean)

# Add Binary Classification Label: IsDelayed
delay_threshold = 15  # Define threshold in minutes for a flight to be considered delayed
df_2023_clean = df_2023_clean.withColumn("IsDelayed", when(col("DelayMinutes") > delay_threshold, 1).otherwise(0))
df_2007_clean = df_2007_clean.withColumn("IsDelayed", when(col("DelayMinutes") > delay_threshold, 1).otherwise(0))

# Prepare features for clustering
assembler_clustering = VectorAssembler(inputCols=["DayOfWeek", "DepHour", "Month", "Distance"], outputCol="features")
data_2023_clustering = assembler_clustering.transform(df_2023_clean).select("features").na.drop()
data_2007_clustering = assembler_clustering.transform(df_2007_clean).select("features").na.drop()

# Train K-Means model
kmeans = KMeans(k=5, seed=42, featuresCol="features", predictionCol="cluster")
model_2023_kmeans = kmeans.fit(data_2023_clustering)
model_2007_kmeans = kmeans.fit(data_2007_clustering)

# Make predictions
predictions_2023_kmeans = model_2023_kmeans.transform(data_2023_clustering)
predictions_2007_kmeans = model_2007_kmeans.transform(data_2007_clustering)

# Evaluate clustering
evaluator_clustering = ClusteringEvaluator(predictionCol="cluster", featuresCol="features", metricName="silhouette")
silhouette_2023 = evaluator_clustering.evaluate(predictions_2023_kmeans)
silhouette_2007 = evaluator_clustering.evaluate(predictions_2007_kmeans)

print(f"2023 K-Means Clustering Silhouette Score: {silhouette_2023}")
print(f"2007 K-Means Clustering Silhouette Score: {silhouette_2007}")

# Prepare features for linear regression
assembler_regression = VectorAssembler(inputCols=["DayOfWeek", "DepHour", "Month", "Distance"], outputCol="features")
data_2023_regression = assembler_regression.transform(df_2023_clean).select("features", "DelayMinutes").na.drop()
data_2007_regression = assembler_regression.transform(df_2007_clean).select("features", "DelayMinutes").na.drop()

# Split data into training and testing sets
train_data_2023, test_data_2023 = data_2023_regression.randomSplit([0.8, 0.2], seed=42)
train_data_2007, test_data_2007 = data_2007_regression.randomSplit([0.8, 0.2], seed=42)

# Linear Regression Model for Delay Prediction
lr_regressor = LinearRegression(featuresCol="features", labelCol="DelayMinutes")
model_2023_lr = lr_regressor.fit(train_data_2023)
model_2007_lr = lr_regressor.fit(train_data_2007)

# Evaluate Linear Regression Model on Test Data
predictions_2023_lr = model_2023_lr.transform(test_data_2023)
predictions_2007_lr = model_2007_lr.transform(test_data_2007)

# Evaluate using RegressionEvaluator
evaluator_regression = RegressionEvaluator(labelCol="DelayMinutes", predictionCol="prediction", metricName="rmse")

rmse_2023 = evaluator_regression.evaluate(predictions_2023_lr)
rmse_2007 = evaluator_regression.evaluate(predictions_2007_lr)

print(f"2023 Linear Regression Model RMSE: {rmse_2023}")
print(f"2007 Linear Regression Model RMSE: {rmse_2007}")

# Compare the number of delays/cancellations between 2007 and 2023
delays_2023 = df_2023_clean.filter(col("IsDelayed") == 1).count()
delays_2007 = df_2007_clean.filter(col("IsDelayed") == 1).count()
cancellations_2023 = df_2023_clean.filter(col("Cancelled") == True).count()
cancellations_2007 = df_2007_clean.filter(col("Cancelled") == True).count()

print(f"Number of Delays in 2023: {delays_2023}")
print(f"Number of Delays in 2007: {delays_2007}")
print(f"Number of Cancellations in 2023: {cancellations_2023}")
print(f"Number of Cancellations in 2007: {cancellations_2007}")

# Predict future flight delays and cancellations
future_data = spark.createDataFrame([
    (1, 10, 6, 500),  # Example data: DayOfWeek, DepHour, Month, Distance
    (5, 15, 12, 1000)
], ["DayOfWeek", "DepHour", "Month", "Distance"])

future_data = assembler_regression.transform(future_data)

# Predict future delays
future_predictions = model_2023_lr.transform(future_data)
future_predictions.show()

# Save clustering and regression results back to HDFS if needed
predictions_2023_kmeans.write.csv(hdfs_host + "/project/output/clustering_2023_predictions.csv", header=True)
predictions_2007_kmeans.write.csv(hdfs_host + "/project/output/clustering_2007_predictions.csv", header=True)
predictions_2023_lr.write.csv(hdfs_host + "/project/output/regression_2023_predictions.csv", header=True)
predictions_2007_lr.write.csv(hdfs_host + "/project/output/regression_2007_predictions.csv", header=True)