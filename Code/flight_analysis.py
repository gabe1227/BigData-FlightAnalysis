from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, hour, month, col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator, BinaryClassificationEvaluator

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

# Prepare features for classification model
assembler_classification = VectorAssembler(inputCols=["DayOfWeek", "DepHour", "Month", "Distance"], outputCol="features")
data_2023_classification = assembler_classification.transform(df_2023_clean).select("features", "IsDelayed").na.drop()
data_2007_classification = assembler_classification.transform(df_2007_clean).select("features", "IsDelayed").na.drop()

# Split data into training and testing sets
train_data_2023, test_data_2023 = data_2023_classification.randomSplit([0.8, 0.2], seed=42)
train_data_2007, test_data_2007 = data_2007_classification.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression Model for Classification
lr_classifier = LogisticRegression(featuresCol="features", labelCol="IsDelayed")
model_2023 = lr_classifier.fit(train_data_2023)
model_2007 = lr_classifier.fit(train_data_2007)

# Evaluate Classification Model on Test Data
predictions_2023 = model_2023.transform(test_data_2023)
predictions_2007 = model_2007.transform(test_data_2007)

# Evaluate using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="IsDelayed", rawPredictionCol="prediction", metricName="areaUnderROC")

roc_2023 = evaluator.evaluate(predictions_2023)
roc_2007 = evaluator.evaluate(predictions_2007)

print(f"2023 Classification Model ROC AUC: {roc_2023}")
print(f"2007 Classification Model ROC AUC: {roc_2007}")

# Additional metrics: Accuracy, Precision, Recall, F1-score
def evaluate_classification(predictions, label_col="IsDelayed"):
    tp = predictions.filter((col("prediction") == 1) & (col(label_col) == 1)).count()
    tn = predictions.filter((col("prediction") == 0) & (col(label_col) == 0)).count()
    fp = predictions.filter((col("prediction") == 1) & (col(label_col) == 0)).count()
    fn = predictions.filter((col("prediction") == 0) & (col(label_col) == 1)).count()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1_score

accuracy_2023, precision_2023, recall_2023, f1_2023 = evaluate_classification(predictions_2023)
accuracy_2007, precision_2007, recall_2007, f1_2007 = evaluate_classification(predictions_2007)

print(f"2023 Classification Model - Accuracy: {accuracy_2023}, Precision: {precision_2023}, Recall: {recall_2023}, F1 Score: {f1_2023}")
print(f"2007 Classification Model - Accuracy: {accuracy_2007}, Precision: {precision_2007}, Recall: {recall_2007}, F1 Score: {f1_2007}")

# Save predictions back to HDFS if needed
predictions_2023.write.csv(hdfs_host + "/project/output/classification_2023_predictions.csv", header=True)
predictions_2007.write.csv(hdfs_host + "/project/output/classification_2007_predictions.csv", header=True)