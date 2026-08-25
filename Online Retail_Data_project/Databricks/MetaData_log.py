# Databricks notebook source
# dbutils.fs.rm("/Volumes/project_1/default/logs/pipeline_log", True)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col
from delta.tables import DeltaTable
 
LOG_PATH = "/Volumes/project_1/default/logs/pipeline_log"
 
if DeltaTable.isDeltaTable(spark, LOG_PATH):
    print("pipeline_log already exists — preserving history, not touching it.")
else:
    print("pipeline_log not found — creating it now.")
    schema = StructType([
        StructField("Batch_ID", StringType(), True),
        StructField("Layer", StringType(), True),
        StructField("Start_Time", TimestampType(), True),
        StructField("End_Time", TimestampType(), True),
        StructField("Records", LongType(), True),
        StructField("Status", StringType(), True),
        StructField("Remarks", StringType(), True)
    ])
 
    metadata_df = spark.createDataFrame([], schema)
 
    (metadata_df.write
     .format("delta")
     .mode("overwrite")
     .save(LOG_PATH))

# COMMAND ----------

pipeline_log = spark.read.format("delta").load(
    "/Volumes/project_1/default/logs/pipeline_log"
)
 
display(pipeline_log.orderBy(col("Start_Time").desc()))