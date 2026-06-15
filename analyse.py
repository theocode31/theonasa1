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


