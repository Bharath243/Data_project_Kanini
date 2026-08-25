# Databricks notebook source
from datetime import datetime

LOG_PATH = "/Volumes/project_1/default/logs/pipeline_log"


def _append_log(batch_id, layer, start_time, end_time, records, status, remarks):
    log_row = spark.createDataFrame(
        [(batch_id, layer, start_time, end_time, records, status, remarks)],
        ["Batch_ID", "Layer", "Start_Time", "End_Time", "Records", "Status", "Remarks"]
    )
    log_row.write.format("delta").mode("append").save(LOG_PATH)


def write_and_log(df, path, layer, batch_id, mode="overwrite", merge_schema=False):
    """
    Use this for a plain DataFrame.write(...) — Bronze, Silver, and the
    straightforward Gold tables (customer, product, sales, revenue_country,
    Analysis_overview).
    """
    start_time = datetime.now()
    status = "SUCCESS"
    remarks = f"{layer} write completed"
    records = 0

    try:
        writer = df.write.format("delta").mode(mode)
        if merge_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.save(path)
        records = df.count()

    except Exception as e:
        status = "FAILED"
        remarks = str(e)[:500]
        raise  

    finally:
        end_time = datetime.now()
        _append_log(batch_id, layer, start_time, end_time, records, status, remarks)

    return records


def merge_and_log(layer, batch_id, target_path, merge_fn, remarks="Merge completed"):
    """
    Use this for Delta MERGE / multi-step SCD operations (product SCD1 merge,
    customer SCD2 close-out + append) where the write doesn't happen via a
    single .write() call.

    merge_fn: a zero-argument function that performs the merge/append itself.
    """
    start_time = datetime.now()
    status = "SUCCESS"
    result_remarks = remarks
    records = 0

    try:
        merge_fn()
        records = spark.read.format("delta").load(target_path).count()

    except Exception as e:
        status = "FAILED"
        result_remarks = str(e)[:500]
        raise

    finally:
        end_time = datetime.now()
        _append_log(batch_id, layer, start_time, end_time, records, status, result_remarks)

    return records

def get_batch_id():
    dbutils.widgets.text("batch_id", "")
    passed_in = dbutils.widgets.get("batch_id")
    return passed_in if passed_in else datetime.now().strftime("%Y%m%d_%H%M%S")