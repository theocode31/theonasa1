import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["JAVA_TOOL_OPTIONS"] = ""  

from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .appName("Firstproject") \
    .config("spark.driver.memory", "4g") \
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
df_rul = df_rul.withColumn("RUL", F.least(F.col("RUL"), F.lit(125)))

print("Apercu avec RUL")
df_rul.show(10)


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

from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol="features", labelCol="RUL", maxIter=100, maxDepth=5, seed=42)
modele_gbt = gbt.fit(df_train)

predictions_gbt = modele_gbt.transform(df_test)
predictions_gbt.select("id_moteur", "cycle", "RUL", "prediction").show(10)

rmse_gbt = evaluateur_rmse.evaluate(predictions_gbt)
mae_gbt  = evaluateur_mae.evaluate(predictions_gbt)
r2_gbt   = evaluateur_r2.evaluate(predictions_gbt)

print("Resultats GBT")
print(f"RMSE : {rmse_gbt:.2f} cycles")
print(f"MAE  : {mae_gbt:.2f} cycles")
print(f"R²   : {r2_gbt:.4f}")

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(featuresCol="features", labelCol="RUL", numTrees=100, maxDepth=5, seed=42)
modele_rf = rf.fit(df_train)

predictions_rf = modele_rf.transform(df_test)

rmse_rf = evaluateur_rmse.evaluate(predictions_rf)
mae_rf  = evaluateur_mae.evaluate(predictions_rf)
r2_rf   = evaluateur_r2.evaluate(predictions_rf)

print("\nComparaison ")
print(f"{'Modele':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print(f"{'LinearRegression':<25} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f}")
print(f"{'GBTRegressor':<25} {rmse_gbt:>8.2f} {mae_gbt:>8.2f} {r2_gbt:>8.4f}")
print(f"{'RandomForest':<25} {rmse_rf:>8.2f} {mae_rf:>8.2f} {r2_rf:>8.4f}")



profondeurs = [3, 5]
iterations  = [50, 100]

print("\nAmelioration GBT22/06/2026")
print(f"{'maxDepth':<12} {'maxIter':<12} {'RMSE':>8} {'MAE':>8} {'R²':>8}")

meilleur_rmse   = float("inf")
meilleurs_params = {}

for depth in profondeurs:
    for n_iter in iterations:
        gbt_tune = GBTRegressor(featuresCol="features", labelCol="RUL",
                                maxDepth=depth, maxIter=n_iter, seed=42)
        modele_tune      = gbt_tune.fit(df_train)
        predictions_tune = modele_tune.transform(df_test)

        r_rmse = evaluateur_rmse.evaluate(predictions_tune)
        r_mae  = evaluateur_mae.evaluate(predictions_tune)
        r_r2   = evaluateur_r2.evaluate(predictions_tune)

        print(f"{depth:<12} {n_iter:<12} {r_rmse:>8.2f} {r_mae:>8.2f} {r_r2:>8.4f}")

        if r_rmse < meilleur_rmse:
            meilleur_rmse    = r_rmse
            meilleurs_params = {"maxDepth": depth, "maxIter": n_iter}

print(f"\nMeilleure combinaison : {meilleurs_params} avec RMSE = {meilleur_rmse:.2f}")
#23/06/26
gbt_final = GBTRegressor(featuresCol="features", labelCol="RUL", maxDepth=3, maxIter=100, seed=42)
predictions_final = gbt_final.fit(df_train).transform(df_test)

df_visu = predictions_final.select("id_moteur", "cycle", "RUL", "prediction").toPandas()
df_visu.columns = ["moteur", "cycle", "RUL_reel", "RUL_predit"]

fig = go.Figure()

for moteur in sorted(df_visu["moteur"].unique()):
    df_m = df_visu[df_visu["moteur"] == moteur].sort_values("cycle")
    fig.add_trace(go.Scatter(x=df_m["cycle"], y=df_m["RUL_reel"],
                             mode="lines", name=f"Reel M{moteur}",
                             line=dict(color="steelblue", width=1), opacity=0.4,
                             visible=(moteur == df_visu["moteur"].unique()[0])))
    fig.add_trace(go.Scatter(x=df_m["cycle"], y=df_m["RUL_predit"],
                             mode="lines", name=f"Predit M{moteur}",
                             line=dict(color="tomato", width=1, dash="dash"), opacity=0.4,
                             visible=(moteur == df_visu["moteur"].unique()[0])))

moteurs_uniques = sorted(df_visu["moteur"].unique())
n_traces = 2
boutons_visu = []
for i, m in enumerate(moteurs_uniques):
    visibilite = [False] * len(moteurs_uniques) * n_traces
    visibilite[i * n_traces]     = True
    visibilite[i * n_traces + 1] = True
    boutons_visu.append(dict(label=f"Moteur {m}", method="update",
                             args=[{"visible": visibilite}, {"title": f"RUL reel vs predit - Moteur {m}"}]))

fig.update_layout(
    title="RUL reel vs predit - GBT optimal",
    xaxis_title="Cycle",
    yaxis_title="RUL (cycles restants)",
    updatemenus=[dict(buttons=boutons_visu, direction="down", x=0.1, y=1.15, showactive=True)],
    showlegend=True,
    height=600
)

fig.write_html("rul_reel_vs_predit.html")
print("Graphique sauvegarder : rul_reel_vs_predit.html")


capteurs_retenus = ["capteur_2", "capteur_3", "capteur_4", "capteur_7", "capteur_8",
                    "capteur_9", "capteur_11", "capteur_12", "capteur_13", "capteur_14",
                    "capteur_15", "capteur_17", "capteur_20", "capteur_21"]

window_glissant = Window.partitionBy("id_moteur").orderBy("cycle").rowsBetween(-9, 0)
#je dis window = fenetre pas pour le systeme d'exploitation
df_lisse = df_rul
for capteur in capteurs_retenus:
    df_lisse = df_lisse.withColumn(f"{capteur}_moy", F.avg(capteur).over(window_glissant))

capteurs_moy = [f"{c}_moy" for c in capteurs_retenus]

assembler_moy = VectorAssembler(inputCols=capteurs_moy, outputCol="features_brutes_moy")
df_lisse      = assembler_moy.transform(df_lisse)

scaler_moy = MinMaxScaler(inputCol="features_brutes_moy", outputCol="features_moy")
df_lisse   = scaler_moy.fit(df_lisse).transform(df_lisse).drop("features_brutes_moy")

df_train_moy, df_test_moy = df_lisse.randomSplit([0.8, 0.2], seed=42)

gbt_moy         = GBTRegressor(featuresCol="features_moy", labelCol="RUL", maxDepth=3, maxIter=100, seed=42)
predictions_moy = gbt_moy.fit(df_train_moy).transform(df_test_moy)

rmse_moy = evaluateur_rmse.evaluate(predictions_moy)
mae_moy  = evaluateur_mae.evaluate(predictions_moy)
r2_moy   = evaluateur_r2.evaluate(predictions_moy)

print("\nComparaison avant / apres moyennes glissantes")
print(f"{'Modele':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print(f"{'GBT optimal (sans lissage)':<35} {meilleur_rmse:>8.2f} {'':>8} {'':>8}")
print(f"{'GBT + moyennes glissantes':<35} {rmse_moy:>8.2f} {mae_moy:>8.2f} {r2_moy:>8.4f}")
