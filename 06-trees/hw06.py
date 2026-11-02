"""Homework 6: tree-based models on the 2026 car fuel-efficiency release."""

import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

TARGET = "fuel_efficiency_mpg"

df = pd.read_csv("data/car_fuel_efficiency_2026.csv")
df = df.fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train[TARGET].values
y_val = df_val[TARGET].values

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.drop(columns=[TARGET]).to_dict(orient="records"))
X_val = dv.transform(df_val.drop(columns=[TARGET]).to_dict(orient="records"))


def feature_name(index):
    name = dv.get_feature_names_out()[index]
    return name.split("=")[0]


# Q1: which feature does a max_depth=1 tree split on?
tree = DecisionTreeRegressor(max_depth=1, random_state=1)
tree.fit(X_train, y_train)
q1 = feature_name(tree.tree_.feature[0])
print("Q1 split feature:", q1)

# Q2-Q4: random forest experiments
def rmse(model):
    return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))

rf10 = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf10.fit(X_train, y_train)
q2 = rmse(rf10)
print("Q2 RMSE:", q2)

scores = {}
for n in [10, 50, 100, 150]:
    model = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    scores[n] = rmse(model)
    print(f"Q3 n_estimators={n}: RMSE={scores[n]:.6f} ({round(scores[n], 3)})")
q3 = min(scores, key=lambda n: round(scores[n], 3))
print("Q3 best n_estimators:", q3)

mean_scores = {}
for depth in [10, 15, 20, 25]:
    depth_scores = []
    for n in [10, 50, 100, 150]:
        model = RandomForestRegressor(
            n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1
        )
        model.fit(X_train, y_train)
        depth_scores.append(rmse(model))
    mean_scores[depth] = float(np.mean(depth_scores))
    print(f"Q4 max_depth={depth}: mean RMSE={mean_scores[depth]:.6f}")
q4 = min(mean_scores, key=mean_scores.get)
print("Q4 best max_depth:", q4)

# Q5: most important feature among the four candidates
importances = dict(zip(dv.get_feature_names_out(), rf10.feature_importances_))
for feature in ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"]:
    print(f"Q5 {feature}: {importances.get(feature, 0.0):.6f}")
q5 = max(
    ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"],
    key=lambda f: importances.get(f, 0.0),
)
print("Q5 most important:", q5)


def train_xgb(eta):
    params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(dv.get_feature_names_out()))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(dv.get_feature_names_out()))
    watchlist = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=watchlist, verbose_eval=False)
    pred = model.predict(dval)
    return float(np.sqrt(mean_squared_error(y_val, pred)))


eta_scores = {eta: train_xgb(eta) for eta in [0.3, 0.1]}
for eta, score in eta_scores.items():
    print(f"Q6 eta={eta}: RMSE={score:.6f}")
q6 = min(eta_scores, key=eta_scores.get)
print("Q6 best eta:", q6)

answers = {"Q1": q1, "Q2": round(q2, 3), "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6}
print(answers)
with open("06-trees/answers.pkl", "wb") as f:
    pickle.dump(answers, f)
