import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, to_timestamp

def preprocessing_gdelt_locally():
    """
    It starts a local Spark job to process a GDELT 2.0 Global Knowledge Graph file.
    """
    
    spark = SparkSession.builder.appName("LocalGDELT").master("local[*]").getOrCreate()
    print("INFO: Spark Session has been created locally.")

    path = "C:/Users/aleks/OneDrive/Pulpit/20251111091500.gkg.csv" # an exemplary file I have chosen to analyze 

    # I have defined column indexes that I am interested in - according to the GKG 2.0 Codebook (Reference: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
    idx = {
        "GKGRECORDID": 0,
        "V2_1DATE": 1,
        "V1LOCATIONS": 9,
        "V1PERSONS": 11,
        "V1ORGANIZATIONS": 13,
        "V1_5TONE": 15
    }

    try:
        df_raw = spark.read.option("delimiter", "\t").option("header", "false").csv(path) # GDELT uses tab (\t) as delimiter
    except Exception as e:
        print(f"ERROR: File cannot be uploaded. Check path: {path}")
        print(e)
        spark.stop()
        sys.exit(1)

    print("INFO: The file has been uploaded. Transformation starting...")

    # here, I have chosen only the columns I am interested in 
    df_chosen = df_raw.select(
        col(f"_c{idx['GKGRECORDID']}").alias("RecordID"),
        col(f"_c{idx['V2_1DATE']}").alias("PublicationDateString"),
        col(f"_c{idx['V1_5TONE']}").alias("ToneDataString"),
        col(f"_c{idx['V1PERSONS']}").alias("Persons"),
        col(f"_c{idx['V1ORGANIZATIONS']}").alias("Organizations"),
        col(f"_c{idx['V1LOCATIONS']}").alias("Locations")
    )

    # cleaning date (string 'YYYYMMDDHHMMSS' as a real timestamp)
    df_timestamp = df_chosen.withColumn(
        "PublicationTimestamp",
        to_timestamp(col("PublicationDateString"), "yyyyMMddHHmmss")
    )

    # I have divided V1.5TONE (string "Tone, Pos, Neg,...") into separate columns
    df_tone = df_timestamp.withColumn(
        "ToneArray", split(col("ToneDataString"), ",")
    ).select(
        col("RecordID"),
        col("PublicationTimestamp"),
        col("Persons"),
        col("Organizations"),
        col("Locations"),
        col("ToneArray").getItem(0).cast("float").alias("Tone"),
        col("ToneArray").getItem(1).cast("float").alias("PositiveScore"),
        col("ToneArray").getItem(2).cast("float").alias("NegativeScore"),
        col("ToneArray").getItem(3).cast("float").alias("Polarity"),
        col("ToneArray").getItem(4).cast("float").alias("ActivityRefDensity"),
        col("ToneArray").getItem(5).cast("float").alias("SelfGroupRefDensity")
    )

    # filtering - here, I have deleted records which do not have any entities or tone
    df_filter = df_tone.filter(
        (col("Persons").isNotNull() | col("Organizations").isNotNull() | col("Locations").isNotNull()) & col("Tone").isNotNull()
    )

    print("INFO: Resulting schema:")
    df_filter.printSchema()

    print("INFO: 10 filtered records:")
    df_filter.show(10, truncate = False)
    
    print(f"INFO: Found {df_filter.count()} correct records.")
    
    spark.stop()


if __name__ == "__main__":
    preprocessing_gdelt_locally()