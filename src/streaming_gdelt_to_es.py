from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, current_timestamp, when
from pyspark.ml import PipelineModel
from elasticsearch import Elasticsearch


MODEL_PATH = "/big-data/hive/warehouse/best_model"
KAFKA_TOPIC = "gdelt-events"
KAFKA_SERVER = "localhost:9092"
ES_HOST = "https://localhost:9200"
ES_INDEX = "gdelt_real_predictions"  
ES_USER = "elastic"
ES_PASS = "jCIDwLC=+xE12BVMqBK5"
#ES_CA_CERT = "/etc/elasticsearch/certs/http_ca.crt"  

def send_to_es(batch_df, batch_id):
    if batch_df.count() == 0:
        return

    print(f"INFO: Processing GDELT batch {batch_id} with {batch_df.count()} records...")
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
                "url": row.url,
                "avg_tone": row.avg_tone,
                "pos_score": row.pos_score,
                "neg_score": row.neg_score,
                "predicted_edits": row.predicted_edits,
                "timestamp": row.timestamp.isoformat()
            }
            
            # send to Elasticsearch
            es.index(index=ES_INDEX, document=doc)

        print(f"SUCCESS: GDELT batch {batch_id} sent to Elasticsearch!")
    except Exception as e:
        print(f"ERROR: Could not send to Elasticsearch: {e}")

def run_gdelt_stream():
    spark = SparkSession.builder \
        .appName("GDELT_streaming_to_Elasticsearch") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # loading model
    print(f"INFO: Loading model from {MODEL_PATH}...")
    try:
        loaded_model = PipelineModel.load(MODEL_PATH)
        print("INFO: Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        return

    # reading from Kafka
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_SERVER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # parsing tab separated values since GDELT GKG 2.0 has \t columns
    df_text = df_raw.select(col("value").cast("string").alias("raw_text"))
    
    df_split = df_text.withColumn("split_cols", split(col("raw_text"), "\t"))

    # extracting only URL (5th column) and tone block (16th column)
    df_extracted = df_split.select(
        col("split_cols").getItem(4).alias("url"), 
        col("split_cols").getItem(15).alias("tone_block")
    )

    # spliting tone block (AvgTone, Pos, Neg...)
    df_tone_parsed = df_extracted.withColumn("tone_split", split(col("tone_block"), ",")) \
        .select(
            col("url"),
            col("tone_split").getItem(0).cast("double").alias("AvgTone"),
            col("tone_split").getItem(1).cast("double").alias("PositiveScore"),
            col("tone_split").getItem(2).cast("double").alias("NegativeScore")
        )

    # removing empty rows 
    df_clean = df_tone_parsed.dropna(subset=["AvgTone", "PositiveScore", "NegativeScore"])

    # model prediction
    prediction_df = loaded_model.transform(df_clean)

    # fix on negative predictions (if model returns < 0, set to 0)
    output_df = prediction_df.select(
        col("url"),
        col("AvgTone").alias("avg_tone"),
        col("PositiveScore").alias("pos_score"),
        col("NegativeScore").alias("neg_score"),
        when(col("prediction") < 0, 0).otherwise(col("prediction")).alias("predicted_edits"),
        current_timestamp().alias("timestamp")
    )

    query = output_df.writeStream \
        .foreachBatch(send_to_es) \
        .outputMode("append") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_gdelt_stream()