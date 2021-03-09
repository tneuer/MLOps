import numpy as np
import pandas as pd

data = pd.read_csv("pokemon-classification (pokemon-preprocessing1).csv")


######################
# New data
######################
# data_drift = data.copy()
# data_drift["HP"] = data_drift["HP"] + 50
# data_drift.to_csv("pokemon_50hp.csv")

######################
# New data
######################
data_drift = data.copy()
data_drift["Speed"] = data_drift["Speed"] + 20
data_drift.to_csv("pokemon_processed_20speed.csv")

######################
# Data drift low importance
######################
# data_drift = data.copy()
# data_drift["HP"] = data_drift["HP"] + 200
# data_drift.to_csv("test-pokemon-drift-low.csv")


######################
# Data drift high importance
######################
# data_drift = data.copy()
# data_drift["Speed"] = data_drift["Speed"] + 100
# data_drift["Sp. Atk"] = data_drift["Sp. Atk"] - 100
# data_drift.to_csv("test-pokemon-drift-high.csv")