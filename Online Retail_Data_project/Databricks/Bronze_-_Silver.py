# Databricks notebook source
# MAGIC %run ./Common_functions

# COMMAND ----------

batch_id = get_batch_id()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *


# COMMAND ----------

bronze_df = (
    spark.read
        .option("header", "true")
        .load("/Volumes/project_1/default/bronze/Bronze_onlineretail/")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Removing Duplicates

# COMMAND ----------

silver_df = bronze_df.dropDuplicates()

# COMMAND ----------

silver_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Handing the null Values

# COMMAND ----------

silver_df = silver_df.filter(
    col("Customer_ID").isNotNull() 
)

# COMMAND ----------

silver_df = silver_df.filter(col("Description").isNotNull())

# COMMAND ----------

silver_df = silver_df.filter(col("Invoice").isNotNull())

# COMMAND ----------

silver_df = (
    silver_df
    .withColumn("Description", trim(col("Description")))
    .withColumn("Country", upper(trim(col("Country"))))
)

# COMMAND ----------

silver_df = silver_df.withColumn("StockCode",upper(trim(col("StockCode"))))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Checking and changeing data types

# COMMAND ----------

silver_df = (silver_df
    .withColumn("InvoiceDate",to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss"))
    .withColumn("Quantity",col("Quantity").cast("int"))
    .withColumn("Price",col("Price").cast("double"))
    .withColumn("Customer_ID",col("Customer_ID").cast("long"))
  
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Validating the value that it contains 0 less

# COMMAND ----------

silver_df = silver_df.filter(col("Quantity") > 0 )

# COMMAND ----------

silver_df = silver_df.filter(col("Price") >= 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Making a new column Revenu with Quantity and price

# COMMAND ----------

silver_df = silver_df.withColumn("Revenue",col("Quantity") * col("Price"))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Seperating of the date in the invoice column for the seperating the data by months,week,year

# COMMAND ----------

silver_df = (silver_df
    .withColumn("InvoiceYear", year("InvoiceDate"))
    .withColumn("InvoiceMonth", month("InvoiceDate"))
    .withColumn("InvoiceDay", dayofmonth("InvoiceDate"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Adding of the time when the silver is changet like metadata

# COMMAND ----------

silver_df = (silver_df.withColumn("silver_processed_time",current_timestamp()))

# COMMAND ----------

display(silver_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Creating of the spark table in the silver volume

# COMMAND ----------

silver_path = "/Volumes/project_1/default/silver/Silver_onlineretail"

write_and_log(silver_df, silver_path, layer="Silver", batch_id=batch_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Making a csv file formet into one file so the humans can see easly where the spark data will be with multiple files

# COMMAND ----------

pdf = silver_df.toPandas()
pdf.to_csv(
    "/Volumes/project_1/default/silver/Silver_onlineretail.csv",
    index=False)