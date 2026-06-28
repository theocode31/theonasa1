import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["JAVA_TOOL_OPTIONS"] = ""

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("NASA_CMAPSS_comparaison") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

nom_colonnes = ["id_moteur", "cycle", "reglage_1", "reglage_2", "reglage_3"] \
             + [f"capteur_{i}" for i in range(1, 22)]

capteurs_exclus = ["capteur_1", "capteur_5", "capteur_6", "capteur_10",
                   "capteur_16", "capteur_18", "capteur_19"]

capteurs_retenus = ["capteur_2", "capteur_3", "capteur_4", "capteur_7", "capteur_8",
                    "capteur_9", "capteur_11", "capteur_12", "capteur_13", "capteur_14",
                    "capteur_15", "capteur_17", "capteur_20", "capteur_21"]

evaluateur_rmse = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="rmse")
evaluateur_mae  = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="mae")
evaluateur_r2   = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="r2")

datasets = ["FD001", "FD002", "FD003", "FD004"]

print(f"\n{'Dataset':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")

for dataset in datasets:

    df_brut = spark.read.csv(
        f"train_{dataset}.txt",
        sep=" ", header=False, inferSchema=True
    )
    df_brut = df_brut.select(df_brut.columns[:26])
    df      = df_brut.toDF(*nom_colonnes)

    window      = Window.partitionBy("id_moteur")
    df = df.withColumn("dernier_cycle", F.max("cycle").over(window))
    df = df.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
    df = df.drop("dernier_cycle")
    df = df.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

    window_glissant = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
    for capteur in capteurs_retenus:
        df = df.withColumn(f"{capteur}_moy", F.avg(capteur).over(window_glissant))

    capteurs_moy = [f"{c}_moy" for c in capteurs_retenus]

    assembler = VectorAssembler(inputCols=capteurs_moy, outputCol="features_brutes")
    df        = assembler.transform(df)
    scaler    = MinMaxScaler(inputCol="features_brutes", outputCol="features")
    df        = scaler.fit(df).transform(df).drop("features_brutes")

    df_train, df_test = df.randomSplit([0.8, 0.2], seed=42)

    gbt         = GBTRegressor(featuresCol="features", labelCol="RUL", maxDepth=3, maxIter=100, seed=42)
    predictions = gbt.fit(df_train).transform(df_test)

    rmse = evaluateur_rmse.evaluate(predictions)
    mae  = evaluateur_mae.evaluate(predictions)
    r2   = evaluateur_r2.evaluate(predictions)

    print(f"{dataset:<10} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")

capteurs_avec_reglages = ["reglage_1", "reglage_2", "reglage_3"] + capteurs_retenus


from pyspark.ml.clustering import KMeans

print(f"\nAvec normalisation par condition operationnelle (KMeans k=6)")
print(f"{'Dataset':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")

for dataset in datasets:

    df_brut = spark.read.csv(f"train_{dataset}.txt", sep=" ", header=False, inferSchema=True)
    df_brut = df_brut.select(df_brut.columns[:26])
    df      = df_brut.toDF(*nom_colonnes)

    window = Window.partitionBy("id_moteur")
    df = df.withColumn("dernier_cycle", F.max("cycle").over(window))
    df = df.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
    df = df.drop("dernier_cycle")
    df = df.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

    assembler_reglages = VectorAssembler(inputCols=["reglage_1", "reglage_2", "reglage_3"],
                                         outputCol="reglages_vec")
    df = assembler_reglages.transform(df)
    kmeans = KMeans(featuresCol="reglages_vec", predictionCol="condition", k=6, seed=42)
    df = kmeans.fit(df).transform(df).drop("reglages_vec")

    window_cond = Window.partitionBy("condition")
    for capteur in capteurs_retenus:
        min_c = F.min(capteur).over(window_cond)
        max_c = F.max(capteur).over(window_cond)
        df = df.withColumn(f"{capteur}_norm",
                           (F.col(capteur) - min_c) / (max_c - min_c + F.lit(1e-8)))

    capteurs_norm = [f"{c}_norm" for c in capteurs_retenus]

    window_glissant = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
    for c in capteurs_norm:
        df = df.withColumn(f"{c}_moy", F.avg(c).over(window_glissant))

    capteurs_finaux = [f"{c}_moy" for c in capteurs_norm]

    assembler = VectorAssembler(inputCols=capteurs_finaux, outputCol="features")
    df        = assembler.transform(df)

    df_train, df_test = df.randomSplit([0.8, 0.2], seed=42)

    gbt         = GBTRegressor(featuresCol="features", labelCol="RUL", maxDepth=3, maxIter=100, seed=42)
    predictions = gbt.fit(df_train).transform(df_test)

    rmse_k = evaluateur_rmse.evaluate(predictions)
    mae_k  = evaluateur_mae.evaluate(predictions)
    r2_k   = evaluateur_r2.evaluate(predictions)

    print(f"{dataset:<10} {rmse_k:>8.2f} {mae_k:>8.2f} {r2_k:>8.4f}")



print(f"\nAvec reglage_1/2/3 comme features")
print(f"{'Dataset':<10} {'RMSE':>8} {'MAE':>8} {'R²':>8}")

for dataset in datasets:

    df_brut = spark.read.csv(
        f"train_{dataset}.txt",
        sep=" ", header=False, inferSchema=True
    )
    df_brut = df_brut.select(df_brut.columns[:26])
    df      = df_brut.toDF(*nom_colonnes)

    window = Window.partitionBy("id_moteur")
    df = df.withColumn("dernier_cycle", F.max("cycle").over(window))
    df = df.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
    df = df.drop("dernier_cycle")
    df = df.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

    window_glissant = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
    for capteur in capteurs_avec_reglages:
        df = df.withColumn(f"{capteur}_moy", F.avg(capteur).over(window_glissant))

    capteurs_moy = [f"{c}_moy" for c in capteurs_avec_reglages]

    assembler = VectorAssembler(inputCols=capteurs_moy, outputCol="features_brutes")
    df        = assembler.transform(df)
    scaler    = MinMaxScaler(inputCol="features_brutes", outputCol="features")
    df        = scaler.fit(df).transform(df).drop("features_brutes")

    df_train, df_test = df.randomSplit([0.8, 0.2], seed=42)

    gbt         = GBTRegressor(featuresCol="features", labelCol="RUL", maxDepth=3, maxIter=100, seed=42)
    predictions = gbt.fit(df_train).transform(df_test)

    rmse = evaluateur_rmse.evaluate(predictions)
    mae  = evaluateur_mae.evaluate(predictions)
    r2   = evaluateur_r2.evaluate(predictions)

    print(f"{dataset:<10} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")
