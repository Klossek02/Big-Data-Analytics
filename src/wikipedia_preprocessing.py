import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, from_unixtime, to_date
from pyspark.sql.functions import avg


def wikipedia_preprocessing():
    """
    ETL for Wikipedia Edits:

    1. Load raw Parquet files from Hive.
    2. Transform data: convert timestamp, optionally filter by wiki or other columns.
    3. Validate data: check nulls in all columns, basic statistics.
    4. Save results to silver layer, partitioned by event_date.
    """

    APP_NAME = "Wikipedia_Preprocessing"
    INPUT_PATH = "/big-data/hive/warehouse/wikipedia_edits/"
    OUTPUT_PATH = "/big-data/hive/warehouse/wikipedia_silver/"

    spark = SparkSession.builder \
        .appName(APP_NAME) \
        .enableHiveSupport() \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # loading data
    try:
        # Load raw Parquet
        df_raw = spark.read.format("parquet").load(INPUT_PATH)

        # Convert 'event_timestamp' (unix long) to timestamp
        df_transformed = df_raw.withColumn("EventTimestamp", to_timestamp(from_unixtime(col("event_timestamp")))) \
                               .withColumn("event_date", to_date(from_unixtime(col("event_timestamp"))))

        # Filter only articles (namespace=0) and edits (type='edit')
        df_filtered = df_transformed.filter(
            (col("namespace") == 0) &
            (col("type") == "edit")
        )

        # Selecting only necessary columns
        selected_cols = [
            "id", "title", "page_url", "comment", "EventTimestamp",
            "minor", "bot", "username", "length_old", "length_new", "event_date"
        ]
        df_selected = df_filtered.select(*selected_cols)

        # Preview data
        row_count = df_selected.count()
        if row_count > 0:
            print(f"Total records after filtering: {row_count}")
            print("Top 5 records:")
            df_selected.show(5, truncate=False)
        else:
            print("WARNING: No records to save after filtering.")

        # Check nulls for selected columns
        print("Null value summary for selected columns:")
        null_summary = []
        for c in selected_cols:
            null_count = df_selected.filter(col(c).isNull()).count()
            print(f"  {c}: {null_count}")
            null_summary.append((c, null_count))

        # Count unique ids
        unique_ids = df_selected.select("id").distinct().count()
        total_rows = df_selected.count()
        print(f"Unique IDs: {unique_ids}, Total rows: {total_rows}")

        # Basic metrics
        avg_length_diff = df_filtered.select(avg(col("length_new") - col("length_old"))).collect()[0][0]
        print(f"Average length difference (length_new - length_old): {avg_length_diff:.2f}")
        minor_percentage = df_selected.filter(col("minor") == True).count() / df_selected.count()
        print(f"Percentage of minor edits: {minor_percentage:.2%}")

        # Save to silver layer
        df_selected.write.mode("overwrite").partitionBy("event_date").parquet(OUTPUT_PATH)
        print("SUCCESS: Data saved to silver layer.")

    except Exception as e:
        print("ERROR: Failed to process Wikipedia Edits data.")
        print(str(e))
        sys.exit(1)

    finally:
        spark.stop()

if __name__ == "__main__":

    wikipedia_preprocessing()
