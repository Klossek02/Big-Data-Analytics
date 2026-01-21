import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, trim, lit
from elasticsearch import Elasticsearch, helpers

APP_NAME = "GDELT_Entity_Batch_To_ES"
INPUT_PATH = "/big-data/hive/warehouse/gdelt_silver"

ES_HOST = "https://localhost:9200"
ES_INDEX = "gdelt_batch"
ES_USER = "elastic"
ES_PASS = "jCIDwLC=+xE12BVMqBK5"


def write_partition_to_es(rows):
    """
    Writes a single Spark partition to Elasticsearch using bulk API.
    """
    es = Elasticsearch(
        [ES_HOST],
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        ssl_show_warn=False
    )

    actions = []
    for row in rows:
        doc_id = f"{row.RecordID}_{row.entity_type}_{row.EntityName.replace(' ', '_')}"

        actions.append({
            "_index": ES_INDEX,
            "_id": doc_id,
            "_source": {
                "record_id": row.RecordID,
                "timestamp": row.PublicationTimestamp.isoformat(),
                "entity_name": row.EntityName,
                "entity_type": row.entity_type
            }
        })

    if actions:
        helpers.bulk(es, actions, chunk_size=5000)


def gdelt_entity_extraction():
    spark = (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.sql.shuffle.partitions", "400")
        .enableHiveSupport()
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("INFO: Spark session started")

    try:
        print(f"INFO: Reading silver data from {INPUT_PATH}")
        df = spark.read.parquet(INPUT_PATH)

        df_persons = (
            df
            .filter(col("Persons").isNotNull())
            .withColumn("EntityRaw", explode(split(col("Persons"), ";")))
            .withColumn("EntityName", trim(split(col("EntityRaw"), ",").getItem(1)))
            .select(
                col("RecordID"),
                col("PublicationTimestamp"),
                col("EntityName")
            )
            .filter(col("EntityName").isNotNull() & (col("EntityName") != ""))
            .withColumn("entity_type", lit("person"))
        )

        df_locations = (
            df
            .filter(col("Locations").isNotNull())
            .withColumn("EntityRaw", explode(split(col("Locations"), ";")))
            .withColumn("EntityName", trim(split(col("EntityRaw"), ",").getItem(1)))
            .select(
                col("RecordID"),
                col("PublicationTimestamp"),
                col("EntityName")
            )
            .filter(col("EntityName").isNotNull() & (col("EntityName") != ""))
            .withColumn("entity_type", lit("location"))
        )

        df_entities = df_persons.unionByName(df_locations)

        total = df_entities.count()
        print(f"INFO: Total entities to index: {total}")


        print(f"INFO: Writing to Elasticsearch index [{ES_INDEX}]")
        df_entities.foreachPartition(write_partition_to_es)

        print("SUCCESS: Batch indexed to Elasticsearch")

    except Exception as e:
        print("ERROR: Batch job failed")
        print(str(e))
        sys.exit(1)

    finally:
        spark.stop()
        print("INFO: Spark session stopped")


if __name__ == "__main__":
    gdelt_entity_extraction()
