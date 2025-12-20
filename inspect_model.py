from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, GeneralizedLinearRegressionModel

def inspect_best_model():
    spark = SparkSession.builder \
        .appName("Inspect_best_model") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    MODEL_PATH = "/big-data/hive/warehouse/best_model"

    print("\n" + "="*70)
    print(f"INSPECTING MODEL AT: {MODEL_PATH}")
    print("="*70)

    try:
        loaded_pipeline = PipelineModel.load(MODEL_PATH)

        model = loaded_pipeline.stages[-1]

        raw_type = type(model).__name__

        display_name = raw_type

        if raw_type == "LinearRegressionModel":
            reg_param = model.getRegParam() if model.isSet("regParam") else 0.0
            elastic_net = model.getElasticNetParam() if model.isSet("elasticNetParam") else 0.0

            if reg_param == 0.0:
                display_name = "Linear regression (Standard)"
            elif elastic_net == 1.0:
                display_name = f"Lasso regression (L1) | Reg: {reg_param}"
            elif elastic_net == 0.0:
                display_name = f"Ridge regression (L2) | Reg: {reg_param}"
            else:
                display_name = f"ElasticNet | Alpha: {elastic_net}, Reg: {reg_param}"

        print(f"WINNING MODEL VARIANT: {display_name}")
        print(f"RAW CLASS NAME: {raw_type}")
        print("-" * 70)

        feature_names = ["AvgTone", "PositiveScore", "NegativeScore"]

        # LINEAR LOGIC
        if hasattr(model, "coefficients"):
            print("LINEAR COEFFICIENTS DETAILS:")
            print(f"Intercept (baseline): {model.intercept:.4f}")
            print("\nWeights (impact):")

            coeffs = model.coefficients
            for name, weight in zip(feature_names, coeffs):
             
                status = ""
                if abs(weight) < 0.0001:
                    status = " [ELIMINATED/ZERO]"
                elif weight > 0:
                    status = " [POS (+)]"
                else:
                    status = " [NEG (-)]"

                print(f"  * {name:<15}: {weight:.4f}{status}")

            print("\nINTERPRETATION:")
            print("  (+) Higher value -> MORE edits")
            print("  (-) Higher value -> FEWER edits")
            print("  [ELIMINATED] Feature removed by Lasso (not important)")

        # TREE LOGIC
        elif hasattr(model, "featureImportances"):
            print("TREE ENSEMBLE DETAILS:")

            importances = model.featureImportances
            imp_list = []
            for name, imp in zip(feature_names, importances):
                imp_list.append((name, imp))

            imp_list.sort(key=lambda x: x[1], reverse=True)

            print("\nFeature importance (0-100%):")
            for name, imp in imp_list:
                print(f"  * {name:<15}: {imp:.4f}  ({imp*100:.1f}%)")

            if hasattr(model, "getNumTrees"):
                print(f"\nForest Info: {model.getNumTrees} trees")

        else:
            print("Unknown model structure.")
            print(model)

    except Exception as e:
        print(f"ERROR: Could not inspect model. Reason: {e}")
    print("="*70 + "\n")
    spark.stop()

if __name__ == "__main__":
    inspect_best_model()