# Databricks notebook source
# MAGIC %md
# MAGIC ####Extraction of Table From Azure

# COMMAND ----------

# MAGIC %pip install azure-storage-blob
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO


storage_account_key = "GYWM0XY3Ro0HHJHf77UU4tHA3y6DmQa2RXHEbpflCaJ3T41x+ANOq8kEELdaQbEf+cd4aoKwiyJZ+AStaqawaQ=="
storage_account_name = "sqlonlineretaildatab"
container_name = "sqlrawdata"

conn_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={"sqlonlineretaildatab"};"
    f"AccountKey={"storage_account_key"};"
    f"EndpointSuffix=core.windows.net"
)

blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=sqlonlineretaildatab;AccountKey=GYWM0XY3Ro0HHJHf77UU4tHA3y6DmQa2RXHEbpflCaJ3T41x+ANOq8kEELdaQbEf+cd4aoKwiyJZ+AStaqawaQ==;EndpointSuffix=core.windows.net")
container_client = blob_service_client.get_container_client(container_name)


def load_blob_as_spark_df(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    csv_bytes = blob_client.download_blob().readall()
    pdf = pd.read_csv(StringIO(csv_bytes.decode("utf-8")))
    pdf.to_csv(
    "/Volumes/project_1/default/raw/OnlineRetail.csv",
    index=False)
    return spark.createDataFrame(pdf)

df_s = load_blob_as_spark_df("onlineretail.csv")




# COMMAND ----------

display(df_s)