import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["JAVA_TOOL_OPTIONS"] = ""  

from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("Firstproject") \
    .getOrCreate()

df_brut = spark.read.csv(
    "train_FD001.txt",
    sep=" ",
    header=False,
    inferSchema=True
)

df_brut = df_brut.select(df_brut.columns[:26])

nom_colonnes = ["id_moteur", "cycle", "reglage_1", "reglage_2", "reglage_3"] \
             + [f"capteur_{i}" for i in range(1, 22)]

df_propre = df_brut.toDF(*nom_colonnes)

print("Apercu du DataFrame")
df_propre.show(5)
