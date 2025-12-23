import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, to_timestamp

def gdelt_preprocessing():
    """
    In the following code we perform ETL processing: 

    1. We load raw CSV files from HDFS from /big-data/hive/warehouse/gdelt_bronze. 
    2. We use GDELT 2.0 schema (27 columns) according to the GKG 2.0 Codebook (Reference: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
    3. We clean the data - data formatting, extracting sentiment.
    4. We save the results to the HDFS as a parquet file - as a silver layer.
    """
    
    APP_NAME = "GDELT_Preprocessing"
    INPUT_PATH = "/big-data/hive/warehouse/gdelt_bronze"
    OUTPUT_PATH = "/big-data/hive/warehouse/gdelt_silver"
    
    spark = SparkSession.builder \
        .appName(APP_NAME) \
        .enableHiveSupport() \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    print("INFO: Spark session has been created.")

    # SCHEMA DEFINITION
    # DISCLAIMER: I removed dots from column names. For instance, V2.1DATE --> V21DATE, for Spark compatibility
    gdelt_cols = [
        "GKGRECORDID",                  
        "V21DATE",                      
        "V2SOURCECOLLECTIONIDENTIFIER", 
        "V2SOURCECOMMONNAME",           
        "V2DOCUMENTIDENTIFIER",         
        "V1COUNTS",                     
        "V21COUNTS",                    
        "V1THEMES",                     
        "V2ENHANCEDTHEMES",             
        "V1LOCATIONS",                  
        "V2ENHANCEDLOCATIONS",          
        "V1PERSONS",                    
        "V2ENHANCEDPERSONS",            
        "V1ORGANIZATIONS",              
        "V2ENHANCEDORGANIZATIONS",      
        "V15TONE",                      # here, sentiment
        "V21ENHANCEDDATES",            
        "V2GCAM",                       
        "V21SHARINGIMAGE",              
        "V21RELATEDIMAGES",             
        "V21SOCIALIMAGEEMBEDS",         
        "V21SOCIALVIDEOEMBEDS",         
        "V21QUOTATIONS",                
        "V21ALLNAMES",                  
        "V21AMOUNTS",                   
        "V21TRANSLATIONINFO",           
        "V2EXTRASXML"                   
    ]

    # LOADING DATA  
    try:
        print(f"INFO: Loading data from {INPUT_PATH}...")
        df_raw = spark.read \
            .option("delimiter", "\t") \
            .option("header", "false") \
            .option("inferSchema", "false") \
            .csv(INPUT_PATH)
            
        
        df_silver = df_raw.toDF(*gdelt_cols) # here, we apply the col names to the raw data (_c0, _c1 -> GKGRECORDID, ...)


        col_count = len(df_silver.columns)
        print("-" * 40)
        print(f"INFO: Schema applied.")
        print(f"INFO: Column count detected: {col_count}")
        print("-" * 40)
        # -----------------------------------------------------
        
        # DATA TRANSFORMATION 
        
        # create dataframe: 
        # data: string '20241101014500' -> timestamp
        # tone: string '2.4,-1.2,3.5...' -> to float 
        # we selct and change the names of key cols 
        print("INFO: Transforming data...")
        df_transformed = df_silver \
            .withColumn("PublicationTimestamp", to_timestamp(col("V21DATE"), "yyyyMMddHHmmss")) \
            .withColumn("ToneArray", split(col("V15TONE"), ",")) \
            .select(
                col("GKGRECORDID").alias("RecordID"),
                col("PublicationTimestamp"),
                col("V2SOURCECOMMONNAME").alias("SourceCommonName"),
                col("V2DOCUMENTIDENTIFIER").alias("SourceUrl"),
                col("V2ENHANCEDLOCATIONS").alias("Locations"),
                col("V2ENHANCEDPERSONS").alias("Persons"),
                col("V2ENHANCEDORGANIZATIONS").alias("Organizations"),
                col("V2ENHANCEDTHEMES").alias("Themes"),
                # sentiment 
                col("ToneArray").getItem(0).cast("float").alias("AvgTone"),       # avgerage tone 
                col("ToneArray").getItem(1).cast("float").alias("PositiveScore"), # % positive words
                col("ToneArray").getItem(2).cast("float").alias("NegativeScore"), # % negative words
                col("ToneArray").getItem(3).cast("float").alias("Polarity")       # polarity    
            )

        # FILTERING
        # we remove rows which are empty or corrupted - no date/ NERs
        df_filtered = df_transformed.filter(
            col("PublicationTimestamp").isNotNull() & 
            (col("Persons").isNotNull() | col("Organizations").isNotNull() | col("Locations").isNotNull())
        )

        row_count = df_filtered.count()
        print(f"INFO: Valid records found: {row_count}")
        
        if row_count > 0:
            print("-" * 60)
            print("INFO: SAMPLE RECORD PREVIEW:")
            df_filtered.show(1, vertical=True, truncate=False)
            print("-" * 60)
            
            print(f"INFO: Saving to {OUTPUT_PATH}...")
            df_filtered.write.mode("overwrite").parquet(OUTPUT_PATH)
            
            print("SUCCESS: Data has been saved in silver layer.")
        else:
            print("WARNING: No data to save after preprocessing.")

    except Exception as e:
        print("ERROR: Error in preprocessing.")
        print(str(e))
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    gdelt_preprocessing()