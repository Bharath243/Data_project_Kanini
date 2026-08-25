from datetime import datetime
from airflow.decorators import dag, task
from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
from airflow.providers.microsoft.azure.hooks.wasb import WasbHook
import pandas as pd


@dag(
    schedule="0 10 * * *",
    start_date=datetime(2022, 2, 15),
    catchup=False,
    tags=['MsSql_AzureBlob'],
)
def EL():

    @task()
    def sql_extract():
        try:
            hook = MsSqlHook(mssql_conn_id="MssqlConn")
            sql = """
                SELECT t.name AS table_name
                FROM sys.tables t
                WHERE t.name IN ('onlineretail')
            """
            df = hook.get_pandas_df(sql)
            print(df)
            return df['table_name'].tolist()
        except Exception as e:
            print("Data extract error: " + str(e))

    @task()
    def blob_load(table_list):
        try:
            sql_hook = MsSqlHook(mssql_conn_id="MssqlConn")
            wasb_hook = WasbHook(wasb_conn_id='Azure_Blob_Conn')
            for table_name in table_list:
                df = sql_hook.get_pandas_df(f"SELECT * FROM {table_name};")
                csv_data = df.to_csv(index=False)
                wasb_hook.load_string(
                    string_data=csv_data,
                    container_name='sqlrawdata',
                    blob_name=f'{table_name}.csv',
                    overwrite=True,
                )
                print(f"Uploaded {len(df)} rows from {table_name} to Azure Blob Storage")
        except Exception as e:
            print("Data load error: " + str(e))

    table_list = sql_extract()
    blob_load(table_list)


Azure_extract_and_load = EL()