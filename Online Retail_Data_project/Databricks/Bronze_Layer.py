# Databricks notebook source
# MAGIC %run ./Common_functions
# MAGIC

# COMMAND ----------

batch_id = get_batch_id()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *


# COMMAND ----------

df = spark.read.format('csv').option("header", "true").option("inferSchema", "true").load('/Volumes/project_1/default/raw/OnlineRetail.csv')

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.show(5, False)

# COMMAND ----------

bronze_df = (
    df
    .withColumn("Loading_timestamp", current_timestamp())
    .withColumn("processing_date", current_date())
    .withColumn("source_file", lit("onlineretail.csv"))
    .withColumn("layer", lit("Bronze"))
)

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

bronze_path = "/Volumes/project_1/default/bronze/Bronze_onlineretail/"

write_and_log(bronze_df, bronze_path, layer="Bronze", batch_id=batch_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN project_1.default;