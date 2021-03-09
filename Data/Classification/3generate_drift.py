import numpy as np
import pandas as pd

data = pd.read_csv("./pokemon.csv")
data.drop(["#", "Name", "Generation"], axis=1, inplace=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = data.select_dtypes(include=numerics).columns.values

for numeric_column in numeric_columns:
    rounded_mean = int(data[numeric_column].mean())
    shifted_data = data.copy()
    shifted_data[numeric_column] = data[numeric_column] + rounded_mean
    shifted_data.to_csv("./Drift/pokemon-shift-{}-{}.csv".format(numeric_column.replace(" ", ""), rounded_mean))
