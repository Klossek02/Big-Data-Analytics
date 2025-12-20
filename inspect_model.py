from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, GeneralizedLinearRegressionModel

def inspect_best_model():
    spark = SparkSession.builder \
        .appName("Inspect_best_model") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    MODEL_PATH = "/big-data/hive/warehouse/best_model"
    
    print("\n" + "="*60)
    print(f"INSPECTING MODEL AT: {MODEL_PATH}")
    print("="*60)
    
    try:
        loaded_pipeline = PipelineModel.load(MODEL_PATH)
        
        model = loaded_pipeline.stages[-1]
        
        model_type = type(model).__name__
        print(f"WINNING MODEL TYPE: {model_type}")
        print("-" * 60)
        
        feature_names = ["AvgTone", "PositiveScore", "NegativeScore"]
        
        # Logic for linear models (linear, lasso, GLM) 
        if hasattr(model, "coefficients"):
            print("LINEAR MODEL DETAILS:")
            print(f"Intercept (baseline): {model.intercept:.4f}")
            print("\nCoefficients (weights):")
            
            coeffs = model.coefficients
            for name, weight in zip(feature_names, coeffs):
                direction = "POS (+)" if weight > 0 else "NEG (-)"
                print(f"  * {name:<15}: {weight:.4f}  [{direction}]")
                
            print("\nINTERPRETATION:")
            print("  (+) means higher value leads to MORE edits.")
            print("  (-) means higher value leads to FEWER edits.")

        # Logic for trees (Random forst, GBT, decision tree)
        elif hasattr(model, "featureImportances"):
            print("TREE ENSEMBLE DETAILS:")
            
            importances = model.featureImportances
            
            imp_list = []

            for name, imp in zip(feature_names, importances):
                imp_list.append((name, imp))
            
            imp_list.sort(key=lambda x: x[1], reverse=True)
            
            print("\nFeature importance:")
            for name, imp in imp_list:
                print(f"  * {name:<15}: {imp:.4f}  ({imp*100:.1f}%)")
                
            if hasattr(model, "getNumTrees"):
                print(f"\nForest structure:")
                print(f"  * No of trees: {model.getNumTrees}")
            
            if hasattr(model, "toDebugString"):
                debug_str = model.toDebugString
                print(f"\nTree structure preview:")
                print(debug_str[:300] + " ... [truncated]")

        else:
            print("Unknown model type or generic pipeline.")
            print(model)

    except Exception as e:
        print(f"ERROR: Could not inspect model. Reason: {e}")
    print("="*60 + "\n")
    spark.stop()

if __name__ == "__main__":
    inspect_best_model()