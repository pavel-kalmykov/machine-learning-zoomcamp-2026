import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns

    return np, pd, sns


@app.cell
def _(pd):
    # Preparing the dataset
    df = pd.read_csv("01-intro/car_fuel_efficiency_2026.csv")
    df = df[
        [
            "engine_displacement",
            "horsepower",
            "vehicle_weight",
            "model_year",
            "fuel_efficiency_mpg",
        ]
    ]
    df.head()
    return (df,)


@app.cell
def _(df, sns):
    # EDA
    # Look at the fuel_efficiency_mpg variable. Does it have a long tail?
    sns.histplot(df.fuel_efficiency_mpg)
    return


@app.cell
def _(df, np, pd):
    fuel = df.fuel_efficiency_mpg
    pd.Series(
        {
            "|mean - median| (small -> symmetric)": np.abs(fuel.mean() - fuel.median()),
            "|skewness| (< 0.5 -> aprox. symmetric)": np.abs(fuel.skew()),
        }
    )
    return


@app.cell
def _(df):
    # Question 1
    # There's one column with missing values. What is it?
    column_with_nans = df.columns[df.isnull().any()][0]
    column_with_nans
    return (column_with_nans,)


@app.cell
def _(df):
    # Question 2
    # What's the median (50% percentile) for variable 'horsepower'?
    df.horsepower.median()
    return


@app.cell
def _(df, np, pd):
    # Prepare and split the dataset (lección 2.4) + helpers centralizados
    def split_shuffled(dframe, seed):
        n = len(dframe)
        n_val = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - n_val - n_test
        np.random.seed(seed)
        idx = np.arange(n)
        np.random.shuffle(idx)
        return (
            dframe.iloc[idx[:n_train]],
            dframe.iloc[idx[n_train : n_train + n_val]],
            dframe.iloc[idx[n_train + n_val :]],
        )

    def xy(dframe):
        return dframe.drop(columns=["fuel_efficiency_mpg"]), dframe[
            "fuel_efficiency_mpg"
        ].to_numpy()

    def fill_nans(dframe, column, value):
        return dframe.fillna({column: value})

    def train_linear_regression(X, y, r=0.0):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T @ X + r * np.eye(X.shape[1])
        return np.linalg.inv(XTX) @ X.T @ y

    def rmse(y, y_pred):
        error = y - y_pred
        return float(np.sqrt((error * error).mean()))

    df_train, df_val, df_test = split_shuffled(df, seed=42)
    return (
        df_test,
        df_train,
        df_val,
        fill_nans,
        rmse,
        split_shuffled,
        train_linear_regression,
        xy,
    )


@app.cell
def _(column_with_nans, df_train, df_val, fill_nans, rmse, train_linear_regression, xy):
    # Question 3
    def run_q3():
        df_train_0 = fill_nans(df_train, column_with_nans, 0)
        df_val_0 = fill_nans(df_val, column_with_nans, 0)
        mean_hp = df_train[column_with_nans].mean()
        df_train_m = fill_nans(df_train, column_with_nans, mean_hp)
        df_val_m = fill_nans(df_val, column_with_nans, mean_hp)

        X_train, y_train = xy(df_train_0)
        w = train_linear_regression(X_train, y_train)
        X_val, y_val = xy(df_val_0)
        rmse_zero = round(rmse(y_val, w[0] + X_val.dot(w[1:])), 3)

        X_train_m, y_train_m = xy(df_train_m)
        w_m = train_linear_regression(X_train_m, y_train_m)
        X_val_m, y_val_m = xy(df_val_m)
        rmse_mean = round(rmse(y_val_m, w_m[0] + X_val_m.dot(w_m[1:])), 3)
        return {"RMSE with 0": rmse_zero, "RMSE with mean": rmse_mean, "best": min(rmse_zero, rmse_mean)}

    q3 = run_q3()
    q3
    return


@app.cell
def _(column_with_nans, df_train, df_val, fill_nans, np, rmse, train_linear_regression, xy):
    # Question 4
    def run_q4():
        scores_r = {}
        weights_evidence = {}
        for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
            X_train_r, y_train_r = xy(fill_nans(df_train, column_with_nans, 0))
            X_val_r, y_val_r = xy(fill_nans(df_val, column_with_nans, 0))
            w = train_linear_regression(X_train_r, y_train_r, r)
            scores_r[r] = round(rmse(y_val_r, w[0] + X_val_r.dot(w[1:])), 4)
            if r in (0, 100):
                weights_evidence[r] = [round(w[0], 3)] + [round(v, 3) for v in w[1:]]
        best_r = min(scores_r, key=scores_r.get)
        return scores_r, weights_evidence, best_r

    scores_r, weights_evidence, best_r = run_q4()
    scores_r
    return (weights_evidence, best_r)


@app.cell
def _(scores_r):
    scores_r
    return


@app.cell
def _(column_with_nans, df, fill_nans, np, rmse, split_shuffled, train_linear_regression, xy):
    # Question 5
    def run_q5():
        seed_scores = []
        for seed in range(10):
            dtr, dv, _ = split_shuffled(df, seed)
            X_train, y_train = xy(fill_nans(dtr, column_with_nans, 0))
            w = train_linear_regression(X_train, y_train)
            X_val, y_val = xy(fill_nans(dv, column_with_nans, 0))
            seed_scores.append(rmse(y_val, w[0] + X_val.dot(w[1:])))
        return round(float(np.std(seed_scores)), 3)

    q5 = run_q5()
    q5
    return


@app.cell
def _(column_with_nans, df, fill_nans, pd, rmse, split_shuffled, train_linear_regression, xy):
    # Question 6
    def run_q6():
        df_train9, df_val9, df_test9 = split_shuffled(df, seed=9)
        df_full = pd.concat([df_train9, df_val9])
        X_full, y_full = xy(fill_nans(df_full, column_with_nans, 0))
        w = train_linear_regression(X_full, y_full, r=0.001)
        X_test, y_test = xy(fill_nans(df_test9, column_with_nans, 0))
        return round(rmse(y_test, w[0] + X_test.dot(w[1:])), 3)

    q6 = run_q6()
    q6
    return


if __name__ == "__main__":
    app.run()
