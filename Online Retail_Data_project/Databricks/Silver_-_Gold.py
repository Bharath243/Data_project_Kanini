# Databricks notebook source
# MAGIC %run ./Common_functions

# COMMAND ----------

batch_id = get_batch_id()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *


# COMMAND ----------

silver_df = (
    spark.read
        .option("header", "true")
        .load("/Volumes/project_1/default/silver/Silver_onlineretail/")
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating customer table

# COMMAND ----------

customer = (
    silver_df.groupBy("Customer_ID", "Country")
    .agg(
        min("InvoiceDate").alias("FirstPurchaseDate"),
        max("InvoiceDate").alias("LastPurchaseDate"),
        countDistinct("Invoice").alias("TotalOrders")
    )
)

# COMMAND ----------

write_and_log(customer, "/Volumes/project_1/default/gold/customer", layer="Gold_Customer", batch_id=batch_id)

# COMMAND ----------

pdf = customer.toPandas()
pdf.to_csv(
    "/Volumes/project_1/default/gold/Gold_Customer.csv",
    index=False)

# COMMAND ----------

customer.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Creating of product Table

# COMMAND ----------

product = (silver_df.select("StockCode","Description").dropDuplicates())

# COMMAND ----------

write_and_log(product, "/Volumes/project_1/default/gold/product", layer="Gold_Product", batch_id=batch_id)

# COMMAND ----------

pdf = product.toPandas()
pdf.to_csv(
    "/Volumes/project_1/default/gold/Gold_Product.csv",
    index=False)

# COMMAND ----------

product.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Creating of scales table

# COMMAND ----------

sales = (
    silver_df.select(
        "Invoice",
        "Customer_ID",
        "StockCode",
        "InvoiceDate",
        "Quantity",
        "Price",
        "Revenue"
    )
)

# COMMAND ----------

write_and_log(sales, "/Volumes/project_1/default/gold/sales", layer="Gold_Sales", batch_id=batch_id)

# COMMAND ----------

pdf = sales.toPandas()
pdf.to_csv(
    "/Volumes/project_1/default/gold/Gold_Sales.csv",
    index=False)

# COMMAND ----------

sales.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Country Revenue Table

# COMMAND ----------

revenue = (
    silver_df.groupBy("Country")
    .agg(
        sum("Revenue").alias("TotalRevenue")
    )
)

# COMMAND ----------

write_and_log(revenue, "/Volumes/project_1/default/gold/revenue_country", layer="Gold_Revenue", batch_id=batch_id)

# COMMAND ----------

pdf = revenue.toPandas()
pdf.to_csv(
    "/Volumes/project_1/default/gold/Gold_Revenue.csv",
    index=False)

# COMMAND ----------

revenue.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Analysis Table 

# COMMAND ----------

At = (
    silver_df
    .agg(
        sum("Revenue").alias("TotalRevenue"),
        countDistinct("Invoice").alias("TotalOrders"),
        countDistinct("Customer_ID").alias("TotalCustomers"),
        sum("Quantity").alias("TotalProductsSold"),
        avg("Price").alias("AverageUnitPrice"),
        avg("Revenue").alias("AverageRevenuePerOrder")
    )
)

# COMMAND ----------

write_and_log(At, "/Volumes/project_1/default/gold/Analysis_overview", layer="Gold_Analysis", batch_id=batch_id)

# COMMAND ----------

pdf = At.toPandas()
pdf.to_csv(
    "/Volumes/project_1/default/gold/Gold_Analysis_overview.csv",
    index=False)

# COMMAND ----------

At.display()

# COMMAND ----------

df = spark.read.format('csv').option("header", "true").option("inferSchema", "true").load('/Volumes/project_1/default/gold/Gold_Customer.csv')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE project_1.default.gold_fact_sales
# MAGIC AS SELECT * FROM delta.`/Volumes/project_1/default/gold/fact_sales`;
# MAGIC
# MAGIC CREATE TABLE project_1.default.gold_dim_customer
# MAGIC AS SELECT * FROM delta.`/Volumes/project_1/default/gold/dim_customer_scd2`;
# MAGIC
# MAGIC CREATE TABLE project_1.default.gold_dim_product
# MAGIC AS SELECT * FROM delta.`/Volumes/project_1/default/gold/product`;
# MAGIC
# MAGIC CREATE TABLE project_1.default.gold_dim_date
# MAGIC AS SELECT * FROM delta.`/Volumes/project_1/default/gold/dim_date`;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL project_1.default.gold_fact_sales;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM project_1.default.gold_dim_date LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE project_1.default.gold_dim_customer
# MAGIC AS SELECT * FROM delta.`/Volumes/project_1/default/gold/dim_customer_scd2`;
# MAGIC

# COMMAND ----------

print("=== Original (Volume, Delta) ===")
spark.read.format("delta").load("/Volumes/project_1/default/gold/fact_sales").printSchema()

print("=== New managed table ===")
spark.read.table("project_1.default.gold_fact_sales").printSchema()

# COMMAND ----------

fact = spark.read.table("project_1.default.gold_fact_sales")

fact.select(
    count("*").alias("total_rows"),
    sum(when(col("Customer_SK").isNull(), 1).otherwise(0)).alias("null_customer_sk"),
    sum(when(col("Product_SK").isNull(), 1).otherwise(0)).alias("null_product_sk"),
    sum(when(col("Date_SK").isNull(), 1).otherwise(0)).alias("null_date_sk")
).show()

# COMMAND ----------

fact = spark.read.table("project_1.default.gold_fact_sales")
dim_customer = spark.read.table("project_1.default.gold_dim_customer")

missing_customers = fact.filter(col("Customer_SK").isNull()).select("Customer_ID").distinct()

# Are these customers missing from the dimension entirely, or present but not "current"?
missing_customers.join(dim_customer, "Customer_ID", "left_anti").count()

# COMMAND ----------

dim_customer.join(missing_customers, "Customer_ID").select(
    "Customer_ID", "Country", "Customer_SK", "CurrentFlag", "EffectiveDate", "EndDate"
).orderBy("Customer_ID", "EffectiveDate").show(50, truncate=False)

# COMMAND ----------

display(
    spark.read.format("delta").load("/Volumes/project_1/default/logs/pipeline_log")
    .filter(col("Layer") == "Gold_Customer_SCD2_Update")
    .orderBy(col("Start_Time").desc())
)

# COMMAND ----------

dim = spark.read.table("project_1.default.gold_dim_customer")
has_current = dim.filter(col("CurrentFlag") == "Y").select("Customer_ID").distinct()
all_ids = dim.select("Customer_ID").distinct()
orphaned = all_ids.join(has_current, "Customer_ID", "left_anti")
orphaned.show()

# COMMAND ----------

repair = (
    silver_df.select("Customer_ID", "Country").dropDuplicates(["Customer_ID"])
    .join(orphaned, "Customer_ID")
    .withColumn("EffectiveDate", current_date())
    .withColumn("EndDate", lit(None).cast("date"))
    .withColumn("CurrentFlag", lit("Y"))
    .withColumn(
        "Customer_SK",
        sha2(concat(col("Customer_ID").cast("string"), lit("|"), col("EffectiveDate").cast("string")), 256)
    )
)

repair.write.format("delta").mode("append").save("/Volumes/project_1/default/gold/dim_customer_scd2")

# COMMAND ----------

# ============================================================
# STEP 1 — Find and repair orphaned customers (no CurrentFlag='Y' row)
# ============================================================

# dim = spark.read.table("project_1.default.gold_dim_customer")
# has_current = dim.filter(col("CurrentFlag") == "Y").select("Customer_ID").distinct()
# all_ids = dim.select("Customer_ID").distinct()
# orphaned = all_ids.join(has_current, "Customer_ID", "left_anti")

# print("Orphaned customers found:", orphaned.count())

# repair = (
#     silver_df.select("Customer_ID", "Country").dropDuplicates(["Customer_ID"])
#     .join(orphaned, "Customer_ID")
#     .withColumn("EffectiveDate", current_date())
#     .withColumn("EndDate", lit(None).cast("date"))
#     .withColumn("CurrentFlag", lit("Y"))
#     .withColumn(
#         "Customer_SK",
#         sha2(concat(col("Customer_ID").cast("string"), lit("|"), col("EffectiveDate").cast("string")), 256)
#     )
# )

# repair.write.format("delta").mode("append").save("/Volumes/project_1/default/gold/dim_customer_scd2")

# # COMMAND ----------

# ============================================================
# STEP 2 — Confirm the repair worked (should print 0)
# ============================================================

dim = spark.read.format("delta").load("/Volumes/project_1/default/gold/dim_customer_scd2")
has_current = dim.filter(col("CurrentFlag") == "Y").select("Customer_ID").distinct()
all_ids = dim.select("Customer_ID").distinct()
orphaned_after = all_ids.join(has_current, "Customer_ID", "left_anti")

print("Orphaned customers remaining:", orphaned_after.count())

# # COMMAND ----------

# # ============================================================
# # STEP 3 — Rebuild fact_sales with the now-corrected dimension
# # ============================================================

# dim_product_current_sk = (
#     spark.read.format("delta").load("/Volumes/project_1/default/gold/product")
#     .select("StockCode", "Product_SK")
# )

# dim_customer_current_sk = (
#     spark.read.format("delta").load("/Volumes/project_1/default/gold/dim_customer_scd2")
#     .filter(col("CurrentFlag") == "Y")
#     .select("Customer_ID", "Customer_SK")
# )

# dim_date_sk = (
#     spark.read.format("delta").load("/Volumes/project_1/default/gold/dim_date")
#     .select("FullDate", "Date_SK")
# )

# fact_sales_base = (
#     silver_df
#     .select("Invoice", "Customer_ID", "StockCode", "InvoiceDate", "Quantity", "Price", "Revenue")
#     .withColumn("InvoiceDateOnly", to_date("InvoiceDate"))
# )

# fact_sales = (
#     fact_sales_base
#     .join(dim_customer_current_sk, "Customer_ID", "left")
#     .join(dim_product_current_sk, "StockCode", "left")
#     .join(dim_date_sk, fact_sales_base["InvoiceDateOnly"] == dim_date_sk["FullDate"], "left")
#     .select(
#         "Invoice", "Customer_SK", "Product_SK", "Date_SK",
#         "InvoiceDate", "Quantity", "Price", "Revenue",
#         "Customer_ID", "StockCode"
#     )
# )

# write_and_log(
#     fact_sales,
#     "/Volumes/project_1/default/gold/fact_sales",
#     layer="Gold_FactSales_Star",
#     batch_id=batch_id
# )

# # COMMAND ----------

# # ============================================================
# # STEP 4 — Confirm fact_sales null count is now 0 across the board
# # ============================================================

fact = spark.read.format("delta").load("/Volumes/project_1/default/gold/fact_sales")

fact.select(
    count("*").alias("total_rows"),
    sum(when(col("Customer_SK").isNull(), 1).otherwise(0)).alias("null_customer_sk"),
    sum(when(col("Product_SK").isNull(), 1).otherwise(0)).alias("null_product_sk"),
    sum(when(col("Date_SK").isNull(), 1).otherwise(0)).alias("null_date_sk")
).show()

# # COMMAND ----------

# # ============================================================
# # STEP 5 — Refresh the 4 managed catalog tables for Power BI
# # Run this as a %sql cell (or spark.sql(...) if staying in Python)
# ============================================================

# COMMAND ----------

product = (silver_df.select("StockCode","Description").dropDuplicates())

# COMMAND ----------

from pyspark.sql.window import Window

product_raw = spark.read.format("delta").load("/Volumes/project_1/default/gold/product")

deduped_product = (
    product_raw
    .withColumn("rn", row_number().over(Window.partitionBy("StockCode").orderBy(col("Description"))))
    .filter(col("rn") == 1)
    .drop("rn")
    .withColumn("Product_SK", sha2(col("StockCode"), 256))
)

deduped_product.write.format("delta").mode("overwrite").save("/Volumes/project_1/default/gold/product")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE project_1.default.gold_dim_product
# MAGIC AS SELECT * FROM delta.`/Volumes/project_1/default/gold/product`;

# COMMAND ----------

product = (silver_df.select("StockCode","Description").dropDuplicates(["StockCode"]))