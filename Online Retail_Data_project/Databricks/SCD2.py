# Databricks notebook source
# MAGIC %run ./Common_functions

# COMMAND ----------

batch_id = get_batch_id()

# COMMAND ----------

dbutils.widgets.dropdown("enable_test_scenario", "No", ["Yes", "No"])
run_test_scenario = dbutils.widgets.get("enable_test_scenario") == "Yes"
 
def apply_test_scenario_product(df):
    if not run_test_scenario:
        return df
    return df.withColumn(
        "Description",
        when(col("StockCode") == "22785", "SQUARE CUSHION COVER RED (TEST)").otherwise(col("Description"))
    )
 
def apply_test_scenario_customer(df):
    if not run_test_scenario:
        return df
    return df.withColumn(
        "Country",
        when(col("Customer_ID") == 13085, "FRANCE (TEST)").otherwise(col("Country"))
    )

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

from delta.tables import DeltaTable

# COMMAND ----------

silver_df = (
    spark.read
        .option("header", "true")
        .load("/Volumes/project_1/default/silver/Silver_onlineretail/")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####SCD TYPE-0

# COMMAND ----------


dim_date_path = "/Volumes/project_1/default/gold/dim_date"
 
def build_dim_date():
    date_range = silver_df.agg(
        min("InvoiceDate").alias("min_date"),
        max("InvoiceDate").alias("max_date")
    ).collect()[0]
 
    dates_df = spark.sql(f"""
        SELECT explode(sequence(
            to_date('{date_range['min_date']}'),
            to_date('{date_range['max_date']}'),
            interval 1 day
        )) AS FullDate
    """)
 
    dim_date = (
        dates_df
        .withColumn("Date_SK", date_format(col("FullDate"), "yyyyMMdd").cast("int"))
        .withColumn("Year", year("FullDate"))
        .withColumn("Month", month("FullDate"))
        .withColumn("Day", dayofmonth("FullDate"))
        .withColumn("Quarter", quarter("FullDate"))
        .withColumn("MonthName", date_format("FullDate", "MMMM"))
        .withColumn("DayName", date_format("FullDate", "EEEE"))
        .withColumn(
            "IsWeekend",
            when(date_format("FullDate", "E").isin("Sat", "Sun"), lit(True)).otherwise(lit(False))
        )
    )
    return dim_date
 
def dim_date_init():
    if DeltaTable.isDeltaTable(spark, dim_date_path):
        print("dim_date already exists — Type 0 means it never gets rebuilt.")
    else:
        print("dim_date not found — building it now.")
        build_dim_date().write.format("delta").mode("overwrite").save(dim_date_path)
 

# COMMAND ----------

merge_and_log(
    "Gold_DimDate_SCD0",
    batch_id,
    dim_date_path,
    dim_date_init,
    remarks="Type 0 — immutable, created once"
)

# COMMAND ----------

display(spark.read.format("delta").load(dim_date_path).limit(5))

# COMMAND ----------

# dbutils.fs.ls("/Volumes/project_1/default/gold/")

# COMMAND ----------

# MAGIC %md
# MAGIC ####SCD TYPE-1

# COMMAND ----------

display(
    silver_df
    .groupBy("StockCode")
    .agg(
        collect_set("Description").alias("Descriptions")
    )
    .withColumn("Description_Count", size("Descriptions"))
    .filter("Description_Count > 1")
)

# COMMAND ----------

gold_product = (
    spark.read
    .format("delta")
    .load("/Volumes/project_1/default/gold/product")
)

display(gold_product)

# COMMAND ----------

display(
    gold_product
    .groupBy("StockCode")
    .agg(
        collect_set("Description").alias("Descriptions")
    )
    .withColumn("Description_Count", size("Descriptions"))
    .filter("Description_Count > 1")
)

# COMMAND ----------

display(
    silver_df
    .filter(col("StockCode")=="22785")
    .select("StockCode","Description")
    .dropDuplicates()
)

# COMMAND ----------


silver_product = apply_test_scenario_product(
    silver_df
    .select("StockCode","Description")
    .dropDuplicates(["StockCode"])
)
 
display(silver_product.filter(col("StockCode")=="22785"))

# COMMAND ----------

# silver_product = (
#     silver_df
#     .select("StockCode","Description")
#     .dropDuplicates(["StockCode"])
#     .withColumn(
#         "Description",
#         when(
#             col("StockCode")=="22785",
#             "SQUARE CUSHION COVER RED"
#         ).otherwise(col("Description"))
#     )
# )

# display(silver_product.filter(col("StockCode")=="22785"))

# COMMAND ----------

silver_product.createOrReplaceTempView("silver_product")

# COMMAND ----------

try:
    spark.sql("ALTER TABLE delta.`/Volumes/project_1/default/gold/product` ADD COLUMNS (Product_SK STRING)")
except Exception:
    pass 
 
spark.sql("""
    UPDATE delta.`/Volumes/project_1/default/gold/product`
    SET Product_SK = sha2(StockCode, 256)
    WHERE Product_SK IS NULL
""")

# COMMAND ----------

def product_merge():
    spark.sql("""
        MERGE INTO delta.`/Volumes/project_1/default/gold/product` AS target
        USING silver_product AS source
        ON target.StockCode = source.StockCode
        WHEN MATCHED AND target.Description <> source.Description
        THEN UPDATE SET target.Description = source.Description
        WHEN NOT MATCHED THEN INSERT (Product_SK, StockCode, Description)
        VALUES (sha2(source.StockCode, 256), source.StockCode, source.Description)
    """)

# COMMAND ----------

merge_and_log(
    "Gold_Product_SCD1",
    batch_id,
    "/Volumes/project_1/default/gold/product",
    product_merge,
    remarks="SCD1 product merge completed"
)

# COMMAND ----------

# def product_merge():
#     spark.sql("""
#         MERGE INTO delta.`/Volumes/project_1/default/gold/product` AS target
#         USING silver_product AS source
#         ON target.StockCode = source.StockCode
#         WHEN MATCHED AND target.Description <> source.Description
#         THEN UPDATE SET target.Description = source.Description
#         WHEN NOT MATCHED THEN INSERT (StockCode, Description)
#         VALUES (source.StockCode, source.Description)
#     """)

# COMMAND ----------

display(
    spark.read
    .format("delta")
    .load("/Volumes/project_1/default/gold/product")
    .filter(col("StockCode")=="22785")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ####SCD TYPE-2

# COMMAND ----------

# gold_customer = (
#     spark.read
#     .format("delta")
#     .load("/Volumes/project_1/default/gold/customer")
# )

# display(gold_customer)

# COMMAND ----------

# gold_customer_scd = (
#     gold_customer
#     .withColumn("EffectiveDate", current_date())
#     .withColumn("EndDate", lit(None).cast("date"))
#     .withColumn("CurrentFlag", lit("Y"))
# )

# write_and_log(
#     gold_customer_scd,
#     "/Volumes/project_1/default/gold/customer",
#     layer="Gold_Customer_Overwrite_Legacy",
#     batch_id=batch_id,
#     mode="overwrite",
#     merge_schema=True
# )

# COMMAND ----------

# silver_customer = (
#     silver_df
#     .select("Customer_ID","Country")
#     .dropDuplicates(["Customer_ID"])
#     .withColumn(
#         "Country",
#         when(
#             col("Customer_ID") == 13085,
#             "FRANCE"
#         ).otherwise(col("Country"))
#     )
# )

# display(silver_customer.filter(col("Customer_ID") == 13085))

# COMMAND ----------

silver_customer = apply_test_scenario_customer(
    silver_df
    .select("Customer_ID","Country")
    .dropDuplicates(["Customer_ID"])
)
 
display(silver_customer.filter(col("Customer_ID") == 13085))

# COMMAND ----------

silver_customer.createOrReplaceTempView("silver_customer")

# COMMAND ----------

spark.read.format("delta").load("/Volumes/project_1/default/gold/customer").printSchema()

# COMMAND ----------

# def customer_scd2_init():
#     if DeltaTable.isDeltaTable(spark, "/Volumes/project_1/default/gold/dim_customer_scd2"):
#         print("dim_customer_scd2 already exists — skipping init, history is preserved.")
#     else:
#         customer_dimension = (
#             silver_df.select("Customer_ID", "Country").dropDuplicates(["Customer_ID"])
#             .withColumn("EffectiveDate", current_date())
#             .withColumn("EndDate", lit(None).cast("date"))
#             .withColumn("CurrentFlag", lit("Y"))
#         )
#         customer_dimension.write.format("delta").mode("overwrite").save(
#             "/Volumes/project_1/default/gold/dim_customer_scd2"
#         )
#     write_and_log(
#     customer_dimension,
#     "/Volumes/project_1/default/gold/dim_customer_scd2",
#     layer="Gold_Customer_SCD2_Init",
#     batch_id=batch_id
# )

# COMMAND ----------

def customer_scd2_init():
    if DeltaTable.isDeltaTable(spark, "/Volumes/project_1/default/gold/dim_customer_scd2"):
        print("dim_customer_scd2 already exists — skipping init, history is preserved.")
    else:
        customer_dimension = (
            silver_df.select("Customer_ID", "Country").dropDuplicates(["Customer_ID"])
            .withColumn("EffectiveDate", current_date())
            .withColumn("EndDate", lit(None).cast("date"))
            .withColumn("CurrentFlag", lit("Y"))
            .withColumn(
                "Customer_SK",
                sha2(concat(col("Customer_ID").cast("string"), lit("|"), col("EffectiveDate").cast("string")), 256)
            )
        )
        customer_dimension.write.format("delta").mode("overwrite").save(
            "/Volumes/project_1/default/gold/dim_customer_scd2"
        )

# COMMAND ----------

customer_dim = spark.read.format("delta").load(
    "/Volumes/project_1/default/gold/dim_customer_scd2"
)

display(customer_dim)

# COMMAND ----------

# silver_customer = (
#     customer_dim
#     .select("Customer_ID", "Country")
#     .withColumn(
#         "Country",
#         when(
#             col("Customer_ID") == 13085,
#             "FRANCE"
#         ).otherwise(col("Country"))
#     )
# )

# display(silver_customer.filter(col("Customer_ID")==13085))

# COMMAND ----------

silver_customer = apply_test_scenario_customer(
    customer_dim.select("Customer_ID", "Country")
)
 
display(silver_customer.filter(col("Customer_ID")==13085))

# COMMAND ----------

# from delta.tables import DeltaTable
# customer_delta = DeltaTable.forPath(
#     spark,
#     "/Volumes/project_1/default/gold/dim_customer_scd2"
# )
 
# changed_customer = (
#     silver_customer.alias("s")
#     .join(
#         customer_dim.alias("t"),
#         col("s.Customer_ID")==col("t.Customer_ID")
#     )
#     .filter(
#         (col("t.CurrentFlag")=="Y") &
#         (col("s.Country")!=col("t.Country"))
#     )
#     .select(
#         col("s.Customer_ID"),
#         col("s.Country")
#     )
# )
 
# new_customer = (
#     changed_customer
#     .withColumn("EffectiveDate", current_date())
#     .withColumn("EndDate", lit(None).cast("date"))
#     .withColumn("CurrentFlag", lit("Y"))
# )

# COMMAND ----------

customer_delta = DeltaTable.forPath(
    spark,
    "/Volumes/project_1/default/gold/dim_customer_scd2"
)
 
changed_customer = (
    silver_customer.alias("s")
    .join(
        customer_dim.alias("t"),
        col("s.Customer_ID")==col("t.Customer_ID")
    )
    .filter(
        (col("t.CurrentFlag")=="Y") &
        (col("s.Country")!=col("t.Country"))
    )
    .select(
        col("s.Customer_ID"),
        col("s.Country")
    )
)
 
new_customer = (
    changed_customer
    .withColumn("EffectiveDate", current_date())
    .withColumn("EndDate", lit(None).cast("date"))
    .withColumn("CurrentFlag", lit("Y"))
    .withColumn(
        "Customer_SK",
        sha2(concat(col("Customer_ID").cast("string"), lit("|"), col("EffectiveDate").cast("string")), 256)
    )
)

# COMMAND ----------

try:
    spark.sql("ALTER TABLE delta.`/Volumes/project_1/default/gold/dim_customer_scd2` ADD COLUMNS (Customer_SK STRING)")
except Exception:
    pass  # column already exists
 
spark.sql("""
    UPDATE delta.`/Volumes/project_1/default/gold/dim_customer_scd2`
    SET Customer_SK = sha2(concat(Customer_ID, '|', EffectiveDate), 256)
    WHERE Customer_SK IS NULL
""")

# COMMAND ----------


# def customer_scd2_update():
#     customer_delta.alias("target").merge(
#         silver_customer.alias("source"),
#         """
#         target.Customer_ID = source.Customer_ID
#         AND target.CurrentFlag = 'Y'
#         """
#     ).whenMatchedUpdate(
#         condition="""
#             target.Country <> source.Country
#         """,
#         set={
#             "EndDate": "current_date()",
#             "CurrentFlag": "'N'"
#         }
#     ).execute()
 
#     new_customer.write \
#         .format("delta") \
#         .mode("append") \
#         .save("/Volumes/project_1/default/gold/dim_customer_scd2")
 
# merge_and_log(
#     "Gold_Customer_SCD2_Update",
#     batch_id,
#     "/Volumes/project_1/default/gold/dim_customer_scd2",
#     customer_scd2_update,
#     remarks="SCD2 close-out and append completed"
# )

# COMMAND ----------

def customer_scd2_update():
    current_dim = spark.read.format("delta").load(
        "/Volumes/project_1/default/gold/dim_customer_scd2"
    ).filter(col("CurrentFlag") == "Y")

    changed_customer = (
        silver_customer.alias("s")
        .join(current_dim.alias("t"), col("s.Customer_ID") == col("t.Customer_ID"))
        .filter(col("s.Country") != col("t.Country"))
        .select(col("s.Customer_ID"), col("s.Country"))
    )

    # NEW: customers that don't exist in the dimension at all yet
    new_arrivals = (
        silver_customer.alias("s")
        .join(current_dim.alias("t"), col("s.Customer_ID") == col("t.Customer_ID"), "left_anti")
        .select("Customer_ID", "Country")
    )

    customer_delta.alias("target").merge(
        changed_customer.alias("source"),
        "target.Customer_ID = source.Customer_ID AND target.CurrentFlag = 'Y'"
    ).whenMatchedUpdate(set={
        "EndDate": "current_date()",
        "CurrentFlag": "'N'"
    }).execute()

    new_rows = (
        changed_customer.unionByName(new_arrivals)
        .withColumn("EffectiveDate", current_date())
        .withColumn("EndDate", lit(None).cast("date"))
        .withColumn("CurrentFlag", lit("Y"))
        .withColumn(
            "Customer_SK",
            sha2(concat(col("Customer_ID").cast("string"), lit("|"), col("EffectiveDate").cast("string")), 256)
        )
    )

    new_rows.write.format("delta").mode("append").save("/Volumes/project_1/default/gold/dim_customer_scd2")

# COMMAND ----------

display(
    spark.read
    .format("delta")
    .load("/Volumes/project_1/default/gold/dim_customer_scd2")
    .filter(col("Customer_ID")==13085)
)

# COMMAND ----------

customer_dim.printSchema()

# COMMAND ----------

display(customer_dim.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ####SCD TYPE-3

# COMMAND ----------


dim_customer_scd3_path = "/Volumes/project_1/default/gold/dim_customer_scd3"
 
silver_customer_dedup = (
    silver_df.select("Customer_ID", "Country").dropDuplicates(["Customer_ID"])
)
 
def customer_scd3_upsert():
    if not DeltaTable.isDeltaTable(spark, dim_customer_scd3_path):
        init_df = (
            silver_customer_dedup
            .withColumn("Previous_Country", lit(None).cast("string"))
            .withColumn("Changed_Date", current_date())
        )
        init_df.write.format("delta").mode("overwrite").save(dim_customer_scd3_path)
        return

    target = DeltaTable.forPath(spark, dim_customer_scd3_path)
    target.alias("target").merge(
        silver_customer_dedup.alias("source"),
        "target.Customer_ID = source.Customer_ID"
    ).whenMatchedUpdate(
        condition="target.Country <> source.Country",
        set={
            "Previous_Country": "target.Country",
            "Country": "source.Country",
            "Changed_Date": "current_date()"
        }
    ).whenNotMatchedInsert(values={
        "Customer_ID": "source.Customer_ID",
        "Country": "source.Country",
        "Previous_Country": "cast(null as string)",
        "Changed_Date": "current_date()"
    }).execute()
 

# COMMAND ----------

merge_and_log(
    "Gold_Customer_SCD3",
    batch_id,
    dim_customer_scd3_path,
    customer_scd3_upsert,
    remarks="Type 3 — overwrite current value, retain only the last previous value"
)

# COMMAND ----------

display(spark.read.format("delta").load(dim_customer_scd3_path).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ####SCD TYPE-4

# COMMAND ----------


# dim_product_current_path = "/Volumes/project_1/default/gold/dim_product_current"
# dim_product_history_path = "/Volumes/project_1/default/gold/dim_product_history"
 
# silver_product_dedup = silver_df.select("StockCode", "Description").dropDuplicates(["StockCode"])
 
# def product_type4_update():
#     snapshot = silver_product_dedup.withColumn("Loaded_At", current_timestamp())
#     snapshot.write.format("delta").mode("append").save(dim_product_history_path)
#     snapshot.write.format("delta").mode("overwrite").save(dim_product_current_path)

# COMMAND ----------

dim_product_current_path = "/Volumes/project_1/default/gold/dim_product_current"
dim_product_history_path = "/Volumes/project_1/default/gold/dim_product_history"
 
silver_product_dedup = silver_df.select("StockCode", "Description").dropDuplicates(["StockCode"])
 
def product_type4_update():
    snapshot = silver_product_dedup.withColumn("Loaded_At", current_timestamp())
 
    if DeltaTable.isDeltaTable(spark, dim_product_history_path):
        latest_per_product = (
            spark.read.format("delta").load(dim_product_history_path)
            .withColumn("rn", row_number().over(Window.partitionBy("StockCode").orderBy(col("Loaded_At").desc())))
            .filter(col("rn") == 1)
            .select("StockCode", col("Description").alias("Last_Description"))
        )
 
        changed_or_new = (
            snapshot.alias("s")
            .join(latest_per_product.alias("h"), "StockCode", "left")
            .filter(col("h.Last_Description").isNull() | (col("s.Description") != col("h.Last_Description")))
            .select("s.StockCode", "s.Description", "s.Loaded_At")
        )
    else:
        changed_or_new = snapshot
 
    changed_or_new.write.format("delta").mode("append").save(dim_product_history_path)
    snapshot.write.format("delta").mode("overwrite").save(dim_product_current_path)
    merge_and_log(
    "Gold_Product_SCD4",
    batch_id,
    dim_product_current_path,
    product_type4_update,
    remarks="Type 4 — current table overwritten, history table appended"
)

# COMMAND ----------

# merge_and_log(
#     "Gold_Product_SCD4",
#     batch_id,
#     dim_product_current_path,
#     product_type4_update,
#     remarks="Type 4 — current table overwritten, history table appended"
# )

# COMMAND ----------

display(spark.read.format("delta").load(dim_product_current_path).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ####SCD TYPE-6

# COMMAND ----------


# dim_customer_scd6_path = "/Volumes/project_1/default/gold/dim_customer_scd6"
 
# def customer_scd6_update():
#     if not DeltaTable.isDeltaTable(spark, dim_customer_scd6_path):
#         init = (
#             silver_customer_dedup
#             .withColumn("Previous_Country", lit(None).cast("string"))
#             .withColumn("EffectiveDate", current_date())
#             .withColumn("EndDate", lit(None).cast("date"))
#             .withColumn("CurrentFlag", lit("Y"))
#         )
#         init.write.format("delta").mode("overwrite").save(dim_customer_scd6_path)
#         return

#     scd6_delta = DeltaTable.forPath(spark, dim_customer_scd6_path)
#     current_dim = spark.read.format("delta").load(dim_customer_scd6_path).filter(col("CurrentFlag") == "Y")
 
#     changed = (
#         silver_customer_dedup.alias("s")
#         .join(current_dim.alias("t"), col("s.Customer_ID") == col("t.Customer_ID"))
#         .filter(col("s.Country") != col("t.Country"))
#         .select(
#             col("s.Customer_ID").alias("Customer_ID"),
#             col("s.Country").alias("New_Country"),
#             col("t.Country").alias("Old_Country")
#         )
#     )
 
#     # Step 1: close out the current row for anyone who changed
#     scd6_delta.alias("target").merge(
#         changed.alias("source"),
#         "target.Customer_ID = source.Customer_ID AND target.CurrentFlag = 'Y'"
#     ).whenMatchedUpdate(set={
#         "EndDate": "current_date()",
#         "CurrentFlag": "'N'"
#     }).execute()
 
#     # Step 2: append the new current row, carrying forward the old value
#     new_rows = (
#         changed
#         .withColumnRenamed("New_Country", "Country")
#         .withColumnRenamed("Old_Country", "Previous_Country")
#         .withColumn("EffectiveDate", current_date())
#         .withColumn("EndDate", lit(None).cast("date"))
#         .withColumn("CurrentFlag", lit("Y"))
#         .select("Customer_ID", "Country", "Previous_Country", "EffectiveDate", "EndDate", "CurrentFlag")
#     )
 
#     new_rows.write.format("delta").mode("append").save(dim_customer_scd6_path)
# merge_and_log(
#     "Gold_Customer_SCD6",
#     batch_id,
#     dim_customer_scd6_path,
#     customer_scd6_update,
#     remarks="Type 6 — hybrid: current flag + full history + previous-value column"
# )
 


# COMMAND ----------

dim_customer_scd6_path = "/Volumes/project_1/default/gold/dim_customer_scd6"
 
def customer_scd6_update():
    if not DeltaTable.isDeltaTable(spark, dim_customer_scd6_path):
        init = (
            silver_customer_dedup
            .withColumn("Previous_Country", lit(None).cast("string"))
            .withColumn("EffectiveDate", current_date())
            .withColumn("EndDate", lit(None).cast("date"))
            .withColumn("CurrentFlag", lit("Y"))
        )
        init.write.format("delta").mode("overwrite").save(dim_customer_scd6_path)
        return
 
    scd6_delta = DeltaTable.forPath(spark, dim_customer_scd6_path)
    current_dim = spark.read.format("delta").load(dim_customer_scd6_path).filter(col("CurrentFlag") == "Y")
 
    changed = (
        silver_customer_dedup.alias("s")
        .join(current_dim.alias("t"), col("s.Customer_ID") == col("t.Customer_ID"))
        .filter(col("s.Country") != col("t.Country"))
        .select(
            col("s.Customer_ID").alias("Customer_ID"),
            col("s.Country").alias("New_Country"),
            col("t.Country").alias("Old_Country")
        )
    )
 
    # Step 1: close out the current row for anyone who changed
    scd6_delta.alias("target").merge(
        changed.alias("source"),
        "target.Customer_ID = source.Customer_ID AND target.CurrentFlag = 'Y'"
    ).whenMatchedUpdate(set={
        "EndDate": "current_date()",
        "CurrentFlag": "'N'"
    }).execute()
 
    # Step 2: append the new current row, carrying forward the old value
    new_rows = (
        changed
        .withColumnRenamed("New_Country", "Country")
        .withColumnRenamed("Old_Country", "Previous_Country")
        .withColumn("EffectiveDate", current_date())
        .withColumn("EndDate", lit(None).cast("date"))
        .withColumn("CurrentFlag", lit("Y"))
        .select("Customer_ID", "Country", "Previous_Country", "EffectiveDate", "EndDate", "CurrentFlag")
    )
 
    new_rows.write.format("delta").mode("append").save(dim_customer_scd6_path)
merge_and_log(
    "Gold_Customer_SCD6",
    batch_id,
    dim_customer_scd6_path,
    customer_scd6_update,
    remarks="Type 6 — hybrid: current flag + full history + previous-value column"
)

# COMMAND ----------

display(spark.read.format("delta").load(dim_customer_scd6_path).orderBy("Customer_ID", "EffectiveDate"))

# COMMAND ----------

fact_sales_base = (
    silver_df
    .select("Invoice", "Customer_ID", "StockCode", "InvoiceDate", "Quantity", "Price", "Revenue")
    .withColumn("InvoiceDateOnly", to_date("InvoiceDate"))
)

# COMMAND ----------

dim_product_current_sk = (
    spark.read.format("delta").load("/Volumes/project_1/default/gold/product")
    .select("StockCode", "Product_SK")
)
 
dim_customer_current_sk = (
    spark.read.format("delta").load("/Volumes/project_1/default/gold/dim_customer_scd2")
    .filter(col("CurrentFlag") == "Y")
    .select("Customer_ID", "Customer_SK")
)
 
dim_date_sk = (
    spark.read.format("delta").load(dim_date_path)
    .select("FullDate", "Date_SK")
)
 
fact_sales = (
    fact_sales_base
    .join(dim_customer_current_sk, "Customer_ID", "left")
    .join(dim_product_current_sk, "StockCode", "left")
    .join(dim_date_sk, fact_sales_base["InvoiceDateOnly"] == dim_date_sk["FullDate"], "left")
    .select(
        "Invoice",
        "Customer_SK",
        "Product_SK",
        "Date_SK",
        "InvoiceDate",
        "Quantity",
        "Price",
        "Revenue",
        "Customer_ID",
        "StockCode"
    )
)
 
write_and_log(
    fact_sales,
    "/Volumes/project_1/default/gold/fact_sales",
    layer="Gold_FactSales_Star",
    batch_id=batch_id
)

# COMMAND ----------

display(spark.read.format("delta").load("/Volumes/project_1/default/gold/fact_sales").limit(5))