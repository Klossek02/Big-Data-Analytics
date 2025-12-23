from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel
from elasticsearch import Elasticsearch


MODEL_PATH = "/big-data/hive/warehouse/best_model"
KAFKA_TOPIC = "gdelt-events"
KAFKA_SERVER = "localhost:9092"
ES_HOST = "http://localhost:9200"  
ES_INDEX = "gdelt_predictions"

def send_to_es(batch_df, batch_id):
    
    if batch_df.count() == 0:
        return
    
    print(f"INFO: Processing Batch {batch_id} with {batch_df.count()} records...")
    
    records = batch_df.collect()
    

    try:
        es = Elasticsearch(ES_HOST)
        
        for row in records:
            
            doc = {
                "url": row.url,
                "avg_tone": row.avg_tone,
                "pos_score": row.pos_score,
                "neg_score": row.neg_score,
                "predicted_edits": row.predicted_edits,
                "timestamp": str(row.timestamp) 
            }
    
            es.index(index=ES_INDEX, document=doc)
            
        print(f"SUCCESS: Batch {batch_id} sent to Elasticsearch!")
    except Exception as e:
        print(f"ERROR: Could not send to ES: {e}")

def run_streaming():
    spark = SparkSession.builder \
        .appName("GDELT_ES_Custom_Sink") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print(f"INFO: Loading model from {MODEL_PATH}...")
    try:
        loaded_model = PipelineModel.load(MODEL_PATH)
        print("INFO: Model loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. Check Hadoop/HDFS connection. {e}")
        return

    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_SERVER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    json_schema = StructType([
        StructField("AvgTone", DoubleType()),
        StructField("PositiveScore", DoubleType()),
        StructField("NegativeScore", DoubleType()),
        StructField("SOURCEURL", StringType())
    ])

    df_parsed = df_raw.select(from_json(col("value").cast("string"), json_schema).alias("data")).select("data.*")
    df_clean = df_parsed.dropna(subset=["AvgTone", "PositiveScore", "NegativeScore"])

    prediction_df = loaded_model.transform(df_clean)

    output_df = prediction_df.select(
        col("SOURCEURL").alias("url"),
        col("AvgTone").alias("avg_tone"),
        col("PositiveScore").alias("pos_score"),
        col("NegativeScore").alias("neg_score"),
        col("prediction").alias("predicted_edits"),
        current_timestamp().alias("timestamp")
    )

    query = output_df.writeStream \
        .foreachBatch(send_to_es) \
        .outputMode("append") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_streaming()