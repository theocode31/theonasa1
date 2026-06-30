import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["JAVA_TOOL_OPTIONS"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from sklearn.preprocessing import MinMaxScaler

# PySpark MLlib na pas de LSTM natif. On utilise PySpark pour charger,
# calculer le RUL, lisser et normaliser. TensorFlow prend ensuite le relai.

spark = SparkSession.builder \
    .appName("NASA_LSTM") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

col_names = ["id_moteur", "cycle", "reglage_1", "reglage_2", "reglage_3"] + \
            [f"capteur_{i}" for i in range(1, 22)]

df_brut = spark.read.csv("train_FD001.txt", sep=" ", header=False, inferSchema=True)
df_brut = df_brut.select(df_brut.columns[:26])
df      = df_brut.toDF(*col_names)

window_moteur = Window.partitionBy("id_moteur")
df = df.withColumn("dernier_cycle", F.max("cycle").over(window_moteur))
df = df.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
df = df.drop("dernier_cycle")
df = df.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

capteurs = ["capteur_2", "capteur_3", "capteur_4", "capteur_7", "capteur_8",
            "capteur_9", "capteur_11", "capteur_12", "capteur_13", "capteur_14",
            "capteur_15", "capteur_17", "capteur_20", "capteur_21"]

window_glissant = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
for capteur in capteurs:
    df = df.withColumn(f"{capteur}_moy", F.avg(capteur).over(window_glissant))

capteurs_moy = [f"{c}_moy" for c in capteurs]

df_pandas = df.select(["id_moteur", "cycle", "RUL"] + capteurs_moy).toPandas()
df_pandas = df_pandas.sort_values(["id_moteur", "cycle"]).reset_index(drop=True)

scaler_np = MinMaxScaler()
df_pandas[capteurs_moy] = scaler_np.fit_transform(df_pandas[capteurs_moy])

print("Preprocessing PySpark termine")
spark.stop()

FENETRE = 30

def creer_sequences(dataframe, fenetre):
    X, y = [], []
    for moteur in dataframe["id_moteur"].unique():
        df_m = dataframe[dataframe["id_moteur"] == moteur]
        vals = df_m[capteurs_moy].values
        rul  = df_m["RUL"].values
        for i in range(fenetre, len(vals)):
            X.append(vals[i - fenetre:i])
            y.append(rul[i])
    return np.array(X), np.array(y)

moteurs = df_pandas["id_moteur"].unique()
moteurs_train, moteurs_test = train_test_split(moteurs, test_size=0.2, random_state=42)

df_train = df_pandas[df_pandas["id_moteur"].isin(moteurs_train)]
df_test  = df_pandas[df_pandas["id_moteur"].isin(moteurs_test)]

X_train, y_train = creer_sequences(df_train, FENETRE)
X_test,  y_test  = creer_sequences(df_test,  FENETRE)

print(f"Train : {X_train.shape} | Test : {X_test.shape}")

modele = Sequential([
    LSTM(64, input_shape=(FENETRE, len(capteurs_moy)), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

modele.compile(optimizer="adam", loss="mse", metrics=["mae"])
modele.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

modele.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

loss, mae = modele.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(loss)

print(f"\nResultats LSTM sur FD001")
print(f"RMSE : {rmse:.2f} cycles")
print(f"MAE  : {mae:.2f} cycles")

print(f"\nComparaison des modeles")
print(f"{'Modele':<35} {'RMSE':>8} {'MAE':>8}")
print(f"{'GBT + moyennes glissantes':<35} {'16.86':>8} {'12.17':>8}")
print(f"{'LSTM (preprocessing PySpark)':<35} {rmse:>8.2f} {mae:>8.2f}")
