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
#01/07/2026
spark2 = SparkSession.builder \
    .appName("NASA_LSTM_enrichi") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df2_brut = spark2.read.csv("train_FD001.txt", sep=" ", header=False, inferSchema=True)
df2_brut = df2_brut.select(df2_brut.columns[:26])
df2      = df2_brut.toDF(*col_names)

window_moteur2 = Window.partitionBy("id_moteur")
df2 = df2.withColumn("dernier_cycle", F.max("cycle").over(window_moteur2))
df2 = df2.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
df2 = df2.drop("dernier_cycle")
df2 = df2.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

window_glissant2 = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
window_lag2      = Window.partitionBy("id_moteur").orderBy("cycle")

for capteur in capteurs:
    df2 = df2.withColumn(f"{capteur}_moy",   F.avg(capteur).over(window_glissant2))
    df2 = df2.withColumn(f"{capteur}_std",   F.stddev(capteur).over(window_glissant2))
    df2 = df2.withColumn(f"{capteur}_delta", F.col(capteur) - F.lag(capteur, 1).over(window_lag2))

capteurs_std   = [f"{c}_std"   for c in capteurs]
capteurs_delta = [f"{c}_delta" for c in capteurs]
toutes_features = capteurs_moy + capteurs_std + capteurs_delta

for col_name in capteurs_std + capteurs_delta:
    df2 = df2.fillna(0, subset=[col_name])

df2_pandas = df2.select(["id_moteur", "cycle", "RUL"] + toutes_features).toPandas()
df2_pandas = df2_pandas.sort_values(["id_moteur", "cycle"]).reset_index(drop=True)

scaler_np2 = MinMaxScaler()
df2_pandas[toutes_features] = scaler_np2.fit_transform(df2_pandas[toutes_features])

print("Preprocessing enrichi (moy + std + delta) termine")
spark2.stop()

def creer_sequences_enr(dataframe, fenetre):
    X, y = [], []
    for moteur in dataframe["id_moteur"].unique():
        df_m = dataframe[dataframe["id_moteur"] == moteur]
        vals = df_m[toutes_features].values
        rul  = df_m["RUL"].values
        for i in range(fenetre, len(vals)):
            X.append(vals[i - fenetre:i])
            y.append(rul[i])
    return np.array(X), np.array(y)

df2_train = df2_pandas[df2_pandas["id_moteur"].isin(moteurs_train)]
df2_test  = df2_pandas[df2_pandas["id_moteur"].isin(moteurs_test)]

X2_train, y2_train = creer_sequences_enr(df2_train, FENETRE)
X2_test,  y2_test  = creer_sequences_enr(df2_test,  FENETRE)

print(f"Train enrichi : {X2_train.shape} | Test enrichi : {X2_test.shape}")

modele_enr = Sequential([
    LSTM(64, input_shape=(FENETRE, len(toutes_features)), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

modele_enr.compile(optimizer="adam", loss="mse", metrics=["mae"])
modele_enr.summary()

early_stop_enr = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

modele_enr.fit(
    X2_train, y2_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop_enr],
    verbose=1
)

loss_enr, mae_enr = modele_enr.evaluate(X2_test, y2_test, verbose=0)
rmse_enr = np.sqrt(loss_enr)

print(f"\nComparaison LSTM features")
print(f"{'Modele':<40} {'RMSE':>8} {'MAE':>8}")
print(f"{'LSTM moy (14 features)':<40} {rmse:>8.2f} {mae:>8.2f}")
print(f"{'LSTM moy+std+delta (42 features)':<40} {rmse_enr:>8.2f} {mae_enr:>8.2f}")

#02/07/2026

datasets = ["FD001", "FD002", "FD003", "FD004"]

gbt_base = {"FD001": 16.86, "FD002": 23.37, "FD003": 15.74, "FD004": 23.57}

print(f"\nLSTM sur tous les datasets")
print(f"{'Dataset':<10} {'RMSE LSTM':>12} {'GBT base':>12} {'Diff':>8}")

spark3 = SparkSession.builder \
    .appName("NASA_LSTM_tous") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

for dataset in datasets:

    df3_brut = spark3.read.csv(f"train_{dataset}.txt", sep=" ", header=False, inferSchema=True)
    df3_brut = df3_brut.select(df3_brut.columns[:26])
    df3      = df3_brut.toDF(*col_names)

    w_moteur = Window.partitionBy("id_moteur")
    df3 = df3.withColumn("dernier_cycle", F.max("cycle").over(w_moteur))
    df3 = df3.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
    df3 = df3.drop("dernier_cycle")
    df3 = df3.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

    w_glissant = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
    for capteur in capteurs:
        df3 = df3.withColumn(f"{capteur}_moy", F.avg(capteur).over(w_glissant))

    df3_pandas = df3.select(["id_moteur", "cycle", "RUL"] + capteurs_moy).toPandas()
    df3_pandas = df3_pandas.sort_values(["id_moteur", "cycle"]).reset_index(drop=True)

    scaler3 = MinMaxScaler()
    df3_pandas[capteurs_moy] = scaler3.fit_transform(df3_pandas[capteurs_moy])

    moteurs3 = df3_pandas["id_moteur"].unique()
    m_train3, m_test3 = train_test_split(moteurs3, test_size=0.2, random_state=42)

    X3_train, y3_train = creer_sequences(df3_pandas[df3_pandas["id_moteur"].isin(m_train3)], FENETRE)
    X3_test,  y3_test  = creer_sequences(df3_pandas[df3_pandas["id_moteur"].isin(m_test3)],  FENETRE)

    modele3 = Sequential([
        LSTM(64, input_shape=(FENETRE, len(capteurs_moy)), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    modele3.compile(optimizer="adam", loss="mse", metrics=["mae"])

    modele3.fit(
        X3_train, y3_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=0
    )

    loss3, _ = modele3.evaluate(X3_test, y3_test, verbose=0)
    rmse3    = np.sqrt(loss3)
    diff     = rmse3 - gbt_base[dataset]

    print(f"{dataset:<10} {rmse3:>12.2f} {gbt_base[dataset]:>12.2f} {diff:>+8.2f}")

spark3.stop()
