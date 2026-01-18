from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object, current_timestamp
from elasticsearch import Elasticsearch


KAFKA_TOPIC = "wikipedia.edits"  
KAFKA_SERVER = "localhost:9092"
ES_HOST = "https://localhost:9200"
ES_INDEX = "wikipedia_real_data"  
ES_USER = "elastic"
ES_PASS = "jCIDwLC=+xE12BVMqBK5"
#ES_CA_CERT = "/etc/elasticsearch/certs/http_ca.crt"  

def send_to_es(batch_df, batch_id):
    if batch_df.count() == 0:
        return
    
    print(f"INFO: Processing Wiki batch {batch_id} with {batch_df.count()} records...")
    records = batch_df.collect()

    try:
        es = Elasticsearch(
            [ES_HOST],
            basic_auth=(ES_USER, ES_PASS),
            verify_certs=False,      
            ssl_show_warn=False
            #ca_certs=ES_CA_CERT
        )
        
        success_count = 0
    
        for row in records:
            doc = {
                "user": row.user,
                "page_title": row.page_title,
                "type": row.type,
                "timestamp": str(row.timestamp)
            }

            # send to Elasticsearch
            es.index(index=ES_INDEX, document=doc)
        
        print(f"SUCCESS: Wiki batch {batch_id} sent to Elasticsearch!")
    except Exception as e:
        print(f"ERROR: Could not send to Elasticsearch: {e}")

def run_wiki_stream():
    spark = SparkSession.builder \
        .appName("Wikipedia_streaming_to_Elasticsearch") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # reading from Kafka
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_SERVER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # parsing data from Kafka (Kafka send bytes, we cast to String)
    df_str = df_raw.select(col("value").cast("string").alias("json_str"))
    
    df_parsed = df_str.select(
        get_json_object(col("json_str"), "$.username").alias("user"),
        get_json_object(col("json_str"), "$.title").alias("page_title"),
        get_json_object(col("json_str"), "$.type").alias("type"),
        current_timestamp().alias("timestamp")
    )

    # filtering out rows with null user or page_title
    df_clean = df_parsed.dropna(subset=["user", "page_title"])


    query = df_clean.writeStream \
        .foreachBatch(send_to_es) \
        .outputMode("append") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_wiki_stream()