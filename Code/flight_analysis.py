from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, hour, month, col
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator

# Initialize Spark session with HDFS configuration
spark = SparkSession.builder \
    .appName("FlightDelayAnalysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://your-hdfs-host:port") \
    .getOrCreate()

# Load the 2023 and 2007 datasets from HDFS
df_2023 = spark.read.option("header", "true").csv("hdfs://your-hdfs-host:port/path/to/flight_delays_2023_cleaned.csv")
df_2007 = spark.read.option("header", "true").csv("hdfs://your-hdfs-host:port/path/to/flight_delays_2007_cleaned.csv")

# Clean data and select relevant columns
def clean_data(df):
    return df.withColumn("DelayMinutes", col("DelayMinutes").cast("double")) \
             .withColumn("Cancelled", col("Cancelled").cast("boolean")) \
             .withColumn("Diverted", col("Diverted").cast("boolean"))

df_2023_clean = clean_data(df_2023)
df_2007_clean = clean_data(df_2007)

# Prepare features for correlation analysis
assembler_corr = VectorAssembler(inputCols=["DepDelay", "ArrDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"], outputCol="features")
vector_data_2007 = assembler_corr.transform(df_2007_clean)
correlation_matrix = Correlation.corr(vector_data_2007, "features").head()[0]
print("Correlation matrix:\n", correlation_matrix)

# Extract day of the week, hour, and month for 2023 data
df_2023_clean = df_2023_clean.withColumn("DayOfWeek", dayofweek("ScheduledDeparture")) \
                             .withColumn("DepHour", hour("ScheduledDeparture")) \
                             .withColumn("Month", month("ScheduledDeparture"))

# Linear Regression Model
assembler_lr = VectorAssembler(inputCols=["DayOfWeek", "DepHour", "Month", "Distance"], outputCol="features")
data_2023_lr = assembler_lr.transform(df_2023_clean).select("features", "DelayMinutes").na.drop()

train_data, test_data = data_2023_lr.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(featuresCol="features", labelCol="DelayMinutes")
lr_model = lr.fit(train_data)

# Evaluate Linear Regression
predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="DelayMinutes", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Linear Regression RMSE: {rmse}")

# K-Means Clustering for Delay Analysis
assembler_kmeans = VectorAssembler(inputCols=["DayOfWeek", "DepHour", "Month", "Distance"], outputCol="features")
data_2023_kmeans = assembler_kmeans.transform(df_2023_clean).select("features", "Cancelled")

kmeans = KMeans(featuresCol="features", k=3, seed=42)
kmeans_model = kmeans.fit(data_2023_kmeans)

# Evaluate K-Means Clustering
predictions_kmeans = kmeans_model.transform(data_2023_kmeans)
evaluator_kmeans = ClusteringEvaluator(predictionCol="prediction", metricName="silhouette")
silhouette_score = evaluator_kmeans.evaluate(predictions_kmeans)
print(f"K-Means Silhouette Score: {silhouette_score}")

# Save cleaned data and predictions back to HDFS if needed
df_2023_clean.write.csv("hdfs://providence:30222/output/cleaned_2023_data.csv", header=True)
df_2007_clean.write.csv("hdfs://providence:port/path/to/cleaned_2007_data.csv", header=True)
predictions.write.csv("hdfs://providence:port/path/to/lr_predictions.csv", header=True)
predictions_kmeans.write.csv("hdfs://providence:port/path/to/kmeans_predictions.csv", header=True)