import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, from_unixtime, to_date, count, lit
from elasticsearch import Elasticsearch, helpers

# ---------------------------
# CONFIG
# ---------------------------
APP_NAME = "Wikipedia_MinorMajor_Aggregation"
INPUT_PATH = "/big-data/hive/warehouse/wikipedia_silver/"
ES_HOST = "https://localhost:9200"
ES_INDEX = "wikipedia_edits_summary"
ES_USER = "elastic"
ES_PASS = "jCIDwLC=+xE12BVMqBK5"
ES_BULK_CHUNK_SIZE = 5000  # increase for faster indexing

# ---------------------------
# FUNCTION TO WRITE PARTITION TO ES
# ---------------------------
def write_partition_to_es(rows):
    es = Elasticsearch(
        [ES_HOST],
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        ssl_show_warn=False
    )

    actions = []
    for row in rows:
        doc_id = f"{row.event_date}_{row.edit_type}"  # unique ID per day and type
        actions.append({
            "_index": ES_INDEX,
            "_id": doc_id,
            "_source": {
                "event_date": row.event_date.isoformat() if hasattr(row.event_date, 'isoformat') else str(row.event_date),
                "edit_type": row.edit_type,
                "edit_count": row.edit_count
            }
        })

    if actions:
        helpers.bulk(es, actions, chunk_size=ES_BULK_CHUNK_SIZE)

# ---------------------------
# MAIN SPARK JOB
# ---------------------------
def wikipedia_minor_major_aggregation():
    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.sql.shuffle.partitions", "100")  # reasonable for single worker
        .enableHiveSupport()
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("INFO: Spark session started")

    try:
        # ---------------------------
        # Load Wikipedia silver layer
        # ---------------------------
        print(f"INFO: Reading Wikipedia edits from {INPUT_PATH}")
        df = spark.read.parquet(INPUT_PATH)

        # Ensure timestamp and date columns
        df = df.withColumn("EventTimestamp", to_timestamp(from_unixtime(col("event_timestamp")))) \
               .withColumn("event_date", to_date(from_unixtime(col("event_timestamp"))))

        # Silver layer already filtered to article edits, no namespace/type needed
        df_filtered = df  # all rows are edits

        # ---------------------------
        # Aggregate minor vs major edits per day
        # ---------------------------
        df_minor = df_filtered.filter(col("minor") == True) \
                              .groupBy("event_date") \
                              .agg(count("*").alias("edit_count")) \
                              .withColumn("edit_type", lit("minor"))

        df_major = df_filtered.filter(col("minor") == False) \
                              .groupBy("event_date") \
                              .agg(count("*").alias("edit_count")) \
                              .withColumn("edit_type", lit("major"))

        df_summary = df_minor.unionByName(df_major).orderBy("event_date")

        total_rows = df_summary.count()
        print(f"INFO: Total aggregated rows to index: {total_rows}")
        df_summary.show(10, truncate=False)

        # ---------------------------
        # Write to Elasticsearch
        # ---------------------------
        print(f"INFO: Writing summary to Elasticsearch index [{ES_INDEX}]")
        df_summary.foreachPartition(write_partition_to_es)

        print("SUCCESS: Wikipedia edits aggregation completed.")

    except Exception as e:
        print("ERROR: Wikipedia aggregation job failed")
        print(str(e))
        sys.exit(1)

    finally:
        spark.stop()
        print("INFO: Spark session stopped")

# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    wikipedia_minor_major_aggregation()
