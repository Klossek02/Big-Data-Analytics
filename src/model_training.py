from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, to_timestamp, date_trunc, count, lit, explode, array, expr
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, GeneralizedLinearRegression, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import sys

def train():
    spark = SparkSession.builder \
        .appName("GDELT_Wiki_train") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print("\n" + "="*80)
    print("INFO: Loading data with SMART AUGMENTATION (Jittering)...")
    print("="*80)

    # GDELT
    try:
        df_gdelt = spark.read.parquet("/big-data/hive/warehouse/gdelt_silver")

        if "PublicationTimestamp" in df_gdelt.columns:
            df_gdelt = df_gdelt.withColumn("event_time", col("PublicationTimestamp").cast("timestamp"))
        elif "SQLDATE" in df_gdelt.columns:
            df_gdelt = df_gdelt.withColumn("event_time", to_timestamp(col("SQLDATE").cast("string"), "yyyyMMddHHmmss"))

        df_gdelt_agg = df_gdelt.groupBy(date_trunc("day", "event_time").alias("time_window")) \
            .agg({"AvgTone": "avg", "PositiveScore": "avg", "NegativeScore": "avg"}) \
            .withColumnRenamed("avg(AvgTone)", "AvgTone") \
            .withColumnRenamed("avg(PositiveScore)", "PositiveScore") \
            .withColumnRenamed("avg(NegativeScore)", "NegativeScore")

    except Exception as e:
        print(f"ERROR GDELT: {e}")
        return

    # WIKIPEDIA 
    df_wiki_agg = None
    try:
        df_wiki = spark.read.parquet("/big-data/hive/warehouse/wikipedia_silver")
        wiki_cols_lower = {c.lower(): c for c in df_wiki.columns}
        wiki_time_col = None
        for p in ["eventtimestamp", "event_timestamp", "timestamp", "date_time"]:
            if p in wiki_cols_lower:
                wiki_time_col = wiki_cols_lower[p]
                break

        if wiki_time_col:
            df_wiki = df_wiki.withColumn("event_time", col(wiki_time_col).cast("timestamp"))
            df_wiki = df_wiki.filter(col("event_time").isNotNull())
            df_wiki_agg = df_wiki.groupBy(date_trunc("day", "event_time").alias("time_window")) \
                .agg(count("*").alias("TargetEdits"))
    except:
        pass

    # JOIN (LEFT JOIN)
    if df_gdelt_agg is not None and df_wiki_agg is not None:
        real_df = df_gdelt_agg.join(df_wiki_agg, on="time_window", how="left")
        real_df = real_df.na.fill({"TargetEdits": 0})


        feature_cols = ["AvgTone", "PositiveScore", "NegativeScore"] # removing features with too many missing values
        real_df = real_df.dropna(subset=feature_cols)
    else:
        print("CRITICAL: Data load failed.")
        return

    base_count = real_df.count()
    print(f"DEBUG: Real base rows: {base_count}")

    # SMART AUGMENTATION (Jittering) 
    # if we have few data, we clone it with small noise to preserve correlation (jiterring)
    target_min_rows = 300

    final_df = real_df

    if base_count < target_min_rows and base_count > 0:
        multiplication_factor = int(target_min_rows / base_count) + 1
        print(f"INFO: Not enough data. Applying jittering (x{multiplication_factor})...")

        augmented_df = real_df.withColumn("dummy", explode(array([lit(x) for x in range(multiplication_factor)])))

        
        augmented_df = augmented_df.withColumn("AvgTone", col("AvgTone") + (rand() - 0.5) * 0.5) \
                                   .withColumn("PositiveScore", col("PositiveScore") + (rand() - 0.5) * 0.5) \
                                   .withColumn("NegativeScore", col("NegativeScore") + (rand() - 0.5) * 0.5) \
                                   .withColumn("TargetEdits", col("TargetEdits") + (rand() - 0.5) * 1.0) \
                                   .drop("dummy")

        augmented_df = augmented_df.withColumn("TargetEdits", expr("CASE WHEN TargetEdits < 0 THEN 0 ELSE TargetEdits END"))

        final_df = augmented_df

    final_count = final_df.count()
    print(f"DEBUG: Final training set size: {final_count}")

    # TRAINING 
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

    assembler = VectorAssembler(inputCols=["AvgTone", "PositiveScore", "NegativeScore"], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

    def get_metrics(predictions):
        evaluator = RegressionEvaluator(labelCol="TargetEdits")
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        return rmse, mse, mae, r2

    results = {}
    models_map = {}

    # models list
    models_to_train = [
        ("Linear regression", LinearRegression(labelCol="TargetEdits", featuresCol="features")),
        ("Lasso regression", LinearRegression(labelCol="TargetEdits", featuresCol="features", elasticNetParam=1.0, regParam=0.1)),
        ("GLM (Gaussian)", GeneralizedLinearRegression(labelCol="TargetEdits", featuresCol="features", family="gaussian")),
        ("Decision tree", DecisionTreeRegressor(labelCol="TargetEdits", featuresCol="features_raw", maxDepth=5)),
        ("Random forest", RandomForestRegressor(labelCol="TargetEdits", featuresCol="features_raw", numTrees=20, maxDepth=5)),
        ("GBT regressor", GBTRegressor(labelCol="TargetEdits", featuresCol="features_raw", maxIter=10))
    ]

    print("\nStarting training...")

    for name, model in models_to_train:
        print(f"... {name}")
        
        if "Tree" in name or "Forest" in name or "GBT" in name:
            stages = [assembler, model]
        else:
            stages = [assembler, scaler, model]

        pipeline = Pipeline(stages=stages)
        try:
            fitted_model = pipeline.fit(train_data)
            preds = fitted_model.transform(test_data)
            results[name] = get_metrics(preds)
            models_map[name] = fitted_model
        except Exception as e:
            print(f"Failed to train {name}: {e}")

 
    print("\n" + "="*105)
    print(f"FINAL RESULTS (Real data + jitter augmentation)")
    print("="*105)
    print(f"| {'Model name':<22} | {'RMSE':<12} | {'MSE':<12} | {'MAE':<12} | {'R2':<12} |")
    print("-" * 105)

    best_name = ""
    best_rmse = float('inf') 
    best_r2 = float('-inf')

    for name, (rmse, mse, mae, r2) in results.items():
        print(f"| {name:<22} | {rmse:<12.2f} | {mse:<12.2f} | {mae:<12.2f} | {r2:<12.4f} |")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    print("="*105)
    print(f"\nWINNER: {best_name.upper()}")

    if best_name:
        model_path = "/big-data/hive/warehouse/best_model"
        models_map[best_name].write().overwrite().save(model_path)
        print(f"SUCCESS: the best model has been saved to {model_path}")

    spark.stop()

if __name__ == "__main__":
    train()