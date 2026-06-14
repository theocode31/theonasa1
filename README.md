# NASA Turbofan Engine Degradation - Analyse PySpark

Analyse du dataset NASA CMAPSS de dégradation de moteurs à réaction, avec PySpark.

## Dataset

Simulation de 100 moteurs turbofan suivis depuis l'état neuf jusqu'à la panne.  
Source : [Kaggle - NASA Turbofan Engine Degradation Simulation](https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation)

### Fichiers
- `train_FD00X.txt` : données d'entraînement (moteurs suivis jusqu'à la panne)
- `test_FD00X.txt` : données de test (moteurs arrêtés avant la panne)

### Structure des données (26 colonnes)
| Colonnes | Description |
|---|---|
| `id_moteur` | Numéro du moteur |
| `cycle` | Numéro du cycle (1 vol = 1 cycle) |
| `reglage_1/2/3` | Conditions opérationnelles |
| `capteur_1` à `capteur_21` | Mesures physiques (températures, pressions, vitesses) |

## Objectif

Prédire la **durée de vie résiduelle** (RUL - Remaining Useful Life) d'un moteur à partir de ses mesures capteurs.

## Stack technique

- Python 3.13
- PySpark 4.1.2
- Java 17 (Temurin)
