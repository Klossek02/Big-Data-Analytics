from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, to_timestamp, window, count, date_trunc
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def train_comparison():
    spark = SparkSession.builder \
        .appName("GDELT_Wiki_Model_Comparison") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print("\nLoading data...")
    
    ###### GDELT ###### 
    try:
        df_gdelt = spark.read.parquet("/big-data/hive/warehouse/gdelt_silver")

        if "PublicationTimestamp" in df_gdelt.columns:
            df_gdelt = df_gdelt.withColumn("event_time", col("PublicationTimestamp").cast("timestamp"))
        elif "SQLDATE" in df_gdelt.columns:
            df_gdelt = df_gdelt.withColumn("event_time", to_timestamp(col("SQLDATE").cast("string"), "yyyyMMddHHmmss"))
        else:
            raise ValueError("No date column in GDELT")

        df_gdelt_agg = df_gdelt.groupBy(date_trunc("hour", "event_time").alias("time_window")) \
            .agg({"AvgTone": "avg", "PositiveScore": "avg", "NegativeScore": "avg"}) \
            .withColumnRenamed("avg(AvgTone)", "AvgTone") \
            .withColumnRenamed("avg(PositiveScore)", "PositiveScore") \
            .withColumnRenamed("avg(NegativeScore)", "NegativeScore")
    except Exception as e:
        print(f"ERROR GDELT: {e}")
        return

    ##### WIKIPEDIA ##### 
    df_wiki_agg = None
    try:
        df_wiki = spark.read.parquet("/big-data/hive/warehouse/wikipedia_silver")
        
        time_cols = ["event_timestamp", "timestamp", "date_time"]
        wiki_time_col = next((c for c in time_cols if c in df_wiki.columns), None)
        
        if wiki_time_col:
            df_wiki = df_wiki.withColumn(wiki_time_col, col(wiki_time_col).cast("timestamp"))
            
            df_wiki_agg = df_wiki.groupBy(date_trunc("hour", wiki_time_col).alias("time_window")) \
                .agg(count("*").alias("TargetEdits"))
    except Exception as e:
        print(f"WARNING Wiki: {e}")


    # JOIN GDELT AND WIKIPEDIA
    used_mock = False
    if df_wiki_agg is not None:
        final_df = df_gdelt_agg.join(df_wiki_agg, on="time_window", how="inner")
        if final_df.count() == 0:
            print("WARNING: No overlapping dates found. Switching to SYNTHETIC data.")
            used_mock = True
    else:
        used_mock = True

    if used_mock:
        # Synthetic data
        final_df = df_gdelt_agg.withColumn("TargetEdits", 
                           (col("AvgTone") * -2.0) + (col("NegativeScore") * 5.0) + rand() * 15)

 

    feature_cols = ["AvgTone", "PositiveScore", "NegativeScore"]
    final_df = final_df.dropna(subset=feature_cols + ["TargetEdits"])
    
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)


    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)


    ############################################
    # MODEL TRAINING
    ############################################
    
    # 1. LINEAR REGRESSION (Baseline) 
    print("\nINFO: Training linear regression - our first model...")
    lr = LinearRegression(labelCol="TargetEdits", featuresCol="features")
 
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
    lr_model = lr_pipeline.fit(train_data)
    lr_preds = lr_model.transform(test_data)

    # 2. RANDOM FOREST (Advanced) 
    print("INFO: Training Random forest - our second model...")
    rf = RandomForestRegressor(labelCol="TargetEdits", featuresCol="features_raw") 
    rf_pipeline = Pipeline(stages=[assembler, rf])

    # small grid to save the memory
    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5]).build()
    
    crossval = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator(labelCol="TargetEdits", metricName="rmse"), numFolds=2)
    
    cv_model = crossval.fit(train_data)
    best_rf_model = cv_model.bestModel
    rf_preds = best_rf_model.transform(test_data)

    ##########################################
    # MODELS EVALUATION
    ##########################################
    def print_metrics(predictions, model_name):
        rmse = RegressionEvaluator(labelCol="TargetEdits", metricName="rmse").evaluate(predictions)
        mse = RegressionEvaluator(labelCol="TargetEdits", metricName="mse").evaluate(predictions)
        mae = RegressionEvaluator(labelCol="TargetEdits", metricName="mae").evaluate(predictions)
        r2 = RegressionEvaluator(labelCol="TargetEdits", metricName="r2").evaluate(predictions)
        
        print(f"| {model_name:<20} | {rmse:.4f} | {mse:.4f} | {mae:.4f} | {r2:.4f} |")

    print("\n" + "="*70)
    print(f"FINAL COMPARISON ({'REAL DATA' if not used_mock else 'SYNTHETIC DATA'})")
    print("="*70)
    print(f"| {'Model Name':<20} | {'RMSE':<6} | {'MSE':<6} | {'MAE':<6} | {'R2':<6} |")
    print("-" * 70)
    print_metrics(lr_preds, "Linear regression")
    print_metrics(rf_preds, "Random forest")
    print("="*70)
    
    
    model_path = "/big-data/hive/warehouse/best_model"
    best_rf_model.write().overwrite().save(model_path)
    print(f"\nSUCCESS: Best model has been saved to {model_path}")
    spark.stop()

if __name__ == "__main__":
    train_comparison()