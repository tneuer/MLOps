import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./pokemon.csv")
data.drop(["#", "Name", "Generation"], axis=1, inplace=True)

least_common = data["Type 1"].value_counts()[-2:].index.values
data = data.loc[(data["Type 1"] != least_common[0]) & ((data["Type 1"] != least_common[1])), :]
data["LogHP"] = np.log(data["HP"])
data["SqrtHP"] = np.sqrt(data["HP"])

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = data.select_dtypes(include=numerics).columns.values
categorical_columns = ["Type 1", "Type 2", "Legendary"]

fig_hist, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 8))
axs = np.ravel(axs)
# axs_hist = data[numeric_columns].hist()
for ax, numeric_col in zip(axs, numeric_columns):
    data[numeric_col].hist(ax=ax)
    ax.set_title(numeric_col)

fig_bar, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axs = np.ravel(axs)
for i, categorical_column in enumerate(categorical_columns):
    data[categorical_column].value_counts().plot.bar(ax=axs[i])

################
# Optional 1
################
from sklearn.preprocessing import StandardScaler, MinMaxScaler

std_scaler = StandardScaler()
std_columns = ["std_"+col for col in numeric_columns]
data[std_columns] = std_scaler.fit_transform(data[numeric_columns])

norm_scaler = MinMaxScaler()
norm_columns = ["norm_"+col for col in numeric_columns]
data[norm_columns] = norm_scaler.fit_transform(data[numeric_columns])

################
# Optional 2
################
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
onehot_columns = ["is_"+type1 for type1 in np.unique(data["Type 1"])]
data[onehot_columns] = onehot_encoder.fit_transform(data["Type 1"].values.reshape(-1, 1))

################
# Split data
################
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=20210308, shuffle=True)

################
# Save data
################
if not os.path.exists("./Processed"):
    os.mkdir("./Processed")
train_data.to_csv("./Processed/pokemon-processed-train.csv", index=False)
test_data.to_csv("./Processed/pokemon-processed-test.csv", index=False)
fig_hist.savefig("./Processed/histograms.png")
fig_bar.savefig("./Processed/barplots.png")
