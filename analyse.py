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
df_propre.show(10)


df_moteur1= df_propre.filter(df_propre.id_moteur==1)
df_p = df_moteur1.select("cycle", "capteur_2").toPandas()


import plotly.express as px
fig = px.line(df_p, x="cycle", y="capteur_2", title="Capteur 2 vs moteur 1")
fig.write_html("capteur_2_moteur_1.html")

import plotly.graph_objects as go

moteurs = df_propre.select("id_moteur").distinct().toPandas()["id_moteur"].tolist()
#la ligne ci dessus permet de recuperer la liste des moteurs DISTINCTS ( uniques ) du dataframe et de la mettre en liste 
fig = go.Figure()# on cree une figure vide

for moteur in moteurs:
    df_m = df_propre.filter(df_propre.id_moteur == moteur).select("cycle", "capteur_2").toPandas()
    fig.add_trace(go.Scatter(x=df_m["cycle"], y=df_m["capteur_2"], mode="lines", name=f"Moteur {moteur}"))

fig.update_layout(title="Capteur 2 pour tous les moteurs", xaxis_title="Cycle", yaxis_title="Valeur")
fig.write_html("capteur2_tous_moteurs.html")

from pyspark.sql import Window
from pyspark.sql import functions as F

window = Window.partitionBy("id_moteur")
df_rul = df_propre.withColumn("dernier_cycle", F.max("cycle").over(window))#on cherche le max d'un moteur 
df_rul = df_rul.withColumn("RUL", F.col("dernier_cycle") - F.col("cycle"))
df_rul = df_rul.drop("dernier_cycle")

print("Apercu avec RUL")
df_rul.show(10)



# Conversion en pandas une seule fois
df_pandas = df_rul.toPandas()

capteurs = [f"capteur_{i}" for i in range(1, 22)]

fig = go.Figure()

for capteur in capteurs:
    for moteur in df_pandas["id_moteur"].unique():
        df_m = df_pandas[df_pandas["id_moteur"] == moteur]
        fig.add_trace(go.Scatter(
            x=df_m["RUL"],
            y=df_m[capteur],
            mode="lines",
            name=f"Moteur {moteur}",
            visible=(capteur == "capteur_1"),
            line=dict(width=1),
            opacity=0.5
        ))

boutons = []
n = len(df_pandas["id_moteur"].unique())
for i, capteur in enumerate(capteurs):
    visibilite = [False] * len(capteurs) * n
    for j in range(n):
        visibilite[i * n + j] = True
    boutons.append(dict(
        label=capteur,
        method="update",
        args=[{"visible": visibilite}, {"title": f"{capteur} en fonction du RUL"}]
    ))

fig.update_layout(
    title="capteur_1 en fonction du RUL",
    xaxis_title="RUL (cycles restants)",
    yaxis_title="Valeur capteur",
    xaxis=dict(autorange="reversed"),
    updatemenus=[dict(buttons=boutons, direction="down", x=0.1, y=1.15, showactive=True)],
    showlegend=False,
    height=600
)

fig.write_html("capteurs_vs_RUL.html")

colonnes_utiles = ["id_moteur", "cycle", "reglage_1", "reglage_2", "reglage_3",
                   "capteur_2", "capteur_3", "capteur_4", "capteur_7", "capteur_8",
                   "capteur_9", "capteur_11", "capteur_12", "capteur_13", "capteur_14",
                   "capteur_15", "capteur_17", "capteur_20", "capteur_21", "RUL"]

df_final = df_rul.select(colonnes_utiles)

print("DataFrame final apres suppression des capteurs inutiles")
df_final.show(5)

from pyspark.ml.feature import VectorAssembler, MinMaxScaler

capteurs_utiles = ["capteur_2", "capteur_3", "capteur_4", "capteur_7", "capteur_8",
                   "capteur_9", "capteur_11", "capteur_12", "capteur_13", "capteur_14",
                   "capteur_15", "capteur_17", "capteur_20", "capteur_21"]

assembler = VectorAssembler(inputCols=capteurs_utiles, outputCol="features_brutes")
df_assemble = assembler.transform(df_final)

scaler = MinMaxScaler(inputCol="features_brutes", outputCol="features")
scaler_model = scaler.fit(df_assemble)
df_normalise = scaler_model.transform(df_assemble)

df_normalise = df_normalise.drop("features_brutes")

print("DataFrame normalise")
df_normalise.select("id_moteur", "cycle", "RUL", "features").show(5, truncate=False)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


df_train, df_test = df_normalise.randomSplit([0.8, 0.2], seed=42)

print(f"Train : {df_train.count()} lignes | Test : {df_test.count()} lignes")


lr = LinearRegression(featuresCol="features", labelCol="RUL", maxIter=100)
modele_lr = lr.fit(df_train)


predictions = modele_lr.transform(df_test)
predictions.select("id_moteur", "cycle", "RUL", "prediction").show(10)


evaluateur_rmse = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="rmse")
evaluateur_mae  = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="mae")
evaluateur_r2   = RegressionEvaluator(labelCol="RUL", predictionCol="prediction", metricName="r2")

rmse = evaluateur_rmse.evaluate(predictions)
mae  = evaluateur_mae.evaluate(predictions)
r2   = evaluateur_r2.evaluate(predictions)

print("Resultats de la regression lineaire")
print(f"RMSE : {rmse:.2f} cycles")
print(f"MAE  : {mae:.2f} cycles")
print(f"R²   : {r2:.4f}")
