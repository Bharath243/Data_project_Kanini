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
# MAGIC ####Data Quality Framework
# MAGIC Every validation step below follows the same pattern instead of silently
# MAGIC filtering: split the DataFrame into "passes" and "fails", tag the fails
# MAGIC with a reason, and keep them in `rejected_frames` to be written to a real
# MAGIC `rejected_records` audit table at the end — instead of just disappearing.

# COMMAND ----------

rejected_frames = []

def capture_rejects(df, reason):
    """Tag failing rows with a reason and keep a copy for the audit table.
    Casts everything to string since this table is for humans to read
    'why was this dropped', not for further computation."""
    tagged = (
        df.select(
            col("Invoice").cast("string"),
            col("StockCode").cast("string"),
            col("Customer_ID").cast("string"),
            col("Description").cast("string"),
            col("Quantity").cast("string"),
            col("Price").cast("string"),
            col("InvoiceDate").cast("string"),
            col("Country").cast("string")
        )
        .withColumn("Rejection_Reason", lit(reason))
        .withColumn("Batch_ID", lit(batch_id))
        .withColumn("Rejected_At", current_timestamp())
    )
    rejected_frames.append(tagged)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Removing Duplicates
# MAGIC Capture what gets removed, not just remove it.

# COMMAND ----------

deduped = bronze_df.dropDuplicates()
duplicate_rows = bronze_df.exceptAll(deduped)

capture_rejects(duplicate_rows, "duplicate_row")

silver_df = deduped

# COMMAND ----------

silver_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Handling the null Values
# MAGIC Each check now captures what it removes before removing it.

# COMMAND ----------

capture_rejects(silver_df.filter(col("Customer_ID").isNull()), "null_customer_id")
silver_df = silver_df.filter(col("Customer_ID").isNotNull())

# COMMAND ----------

capture_rejects(silver_df.filter(col("Description").isNull()), "null_description")
silver_df = silver_df.filter(col("Description").isNotNull())

# COMMAND ----------

capture_rejects(silver_df.filter(col("Invoice").isNull()), "null_invoice")
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
# MAGIC ####Checking and changing data types
# MAGIC NOTE: a bad value silently becomes null on cast (e.g. a non-numeric
# MAGIC Quantity). We check for that right after casting, below, so it still
# MAGIC gets captured as a rejected row instead of quietly vanishing later.

# COMMAND ----------

silver_df = (silver_df
    .withColumn("InvoiceDate",to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss"))
    .withColumn("Quantity",col("Quantity").cast("int"))
    .withColumn("Price",col("Price").cast("double"))
    .withColumn("Customer_ID",col("Customer_ID").cast("long"))
)

# COMMAND ----------

capture_rejects(
    silver_df.filter(col("Quantity").isNull() | col("Price").isNull() | col("InvoiceDate").isNull()),
    "cast_failed"
)
silver_df = silver_df.filter(col("Quantity").isNotNull() & col("Price").isNotNull() & col("InvoiceDate").isNotNull())

# COMMAND ----------

# MAGIC %md
# MAGIC ####Validating the value that it contains 0 less
# MAGIC NOTE: this still drops all cancellation/
# MAGIC return rows (Quantity <= 0) instead of routing them to a separate returns
# MAGIC table (a returns table is a different concept from rejected_records —
# MAGIC these are legitimate business events, not bad data). They ARE captured
# MAGIC in rejected_records below so at least the count/reason is visible now,
# MAGIC which they weren't before.

# COMMAND ----------

capture_rejects(silver_df.filter(col("Quantity") <= 0), "non_positive_quantity")
silver_df = silver_df.filter(col("Quantity") > 0)

# COMMAND ----------

capture_rejects(silver_df.filter(col("Price") < 0), "negative_price")
silver_df = silver_df.filter(col("Price") >= 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Write rejected_records — the audit trail that didn't exist before

# COMMAND ----------

rejected_records = rejected_frames[0]
for frame in rejected_frames[1:]:
    rejected_records = rejected_records.unionByName(frame)

write_and_log(
    rejected_records,
    "/Volumes/project_1/default/silver/rejected_records",
    layer="Silver_RejectedRecords",
    batch_id=batch_id,
    mode="append"
)

# COMMAND ----------

display(
    rejected_records
    .groupBy("Rejection_Reason")
    .count()
    .orderBy(col("count").desc())
)

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