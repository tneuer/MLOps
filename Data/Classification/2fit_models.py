import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import sklearn.metrics as skm

target_col = "Legendary"
drop_cols = [
    'norm_HP', 'norm_Attack', 'norm_Defense', 'norm_Sp. Atk', 'norm_Sp. Def', 'norm_Speed',
    'norm_LogHP', 'norm_SqrtHP', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk',
    'Sp. Def', 'Speed', 'LogHP', 'std_HP', 'std_LogHP'
]
models = {
    "LR": LogisticRegression, "KNN": KNeighborsClassifier, "SVM": SVC,
    "XGB": XGBClassifier, "lightgbm": LGBMClassifier
}
metrics = {
    "Accuracy": skm.accuracy_score, "AUC": skm.roc_auc_score, "F1": skm.f1_score, "LogLoss": skm.log_loss,
    "Recall": skm.recall_score, "Precision": skm.precision_score
}

####################################

data = pd.read_csv("./Processed/pokemon-processed-train.csv")
data.drop(drop_cols, axis=1, inplace=True)

X = data.loc[:, data.columns != target_col]
y = np.ravel(data.loc[:, data.columns == target_col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=20210308, shuffle=False)

result_dict = {key: [] for key in metrics.keys()}
result_dict["Model"] = []
for model_name, model in models.items():
    m = model()
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    for metric_name, metric in metrics.items():
        try:
            result_dict[metric_name].append(metric(y_pred, y_test))
        except ValueError:
            result_dict[metric_name].append(np.nan)
    result_dict["Model"].append(model_name)

result_df = pd.DataFrame(result_dict).set_index("Model").sort_values(by="AUC", ascending=False)
print(result_df.to_markdown(index=True))
