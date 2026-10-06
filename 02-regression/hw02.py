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
def _(df, pd):
    fuel = df.fuel_efficiency_mpg
    pd.DataFrame(
        {
            "valor": [abs(fuel.mean() - fuel.median()), fuel.skew()],
        },
        index=[
            "|mean - median| (small -> symmetric)",
            "|skewness| (< 0.5 -> aprox. symmetric)",
        ],
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
    # Prepare and split the dataset
    def split_shuffled(
        df: pd.DataFrame, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        n_val = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - n_val - n_test

        np.random.seed(seed)
        idx = np.arange(n)
        np.random.shuffle(idx)

        df_train = df.iloc[idx[:n_train]]
        df_val = df.iloc[idx[n_train : n_train + n_val]]
        df_test = df.iloc[idx[n_train + n_val :]]

        return df_train, df_val, df_test

    def xy(dframe, column, fill):
        d = dframe.fillna({column: fill})
        return d.drop(columns="fuel_efficiency_mpg"), d[column]

    df_train, df_val, df_test = split_shuffled(df, seed=42)
    return df_train, df_val, split_shuffled, xy


@app.cell
def _(column_with_nans, df_train, df_val, np, pd, xy):
    # Question 3
    # We need to deal with missing values for the column from Q1.
    # We have two options: fill it with 0 or with the mean of this variable.
    # Try both options. For each, train a linear regression model without regularization using the code from the lessons.
    # For computing the mean, use the training only!
    # Use the validation dataset to evaluate the models and compare the RMSE of each option.
    # Round the RMSE scores to 3 decimal digits using round(score, 3). This keeps the imputation difference visible in this release.
    # Which option gives better RMSE?
    def train_linear_regression(X, y, r=0.0):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X) + r * np.eye(X.shape[1])
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv.dot(X.T).dot(y)

        return w_full[0], w_full[1:]

    def rmse(y, y_pred):
        se = (y - y_pred) ** 2
        mse = se.mean()
        return np.sqrt(mse)

    mean_hp = df_train[column_with_nans].mean()
    X_train, y_train = xy(df_train, column_with_nans, 0)
    X_val, y_val = xy(df_val, column_with_nans, 0)
    X_zero_train, _ = xy(df_train, column_with_nans, 0)
    X_zero_val, _ = xy(df_val, column_with_nans, 0)
    X_mean_train, _ = xy(df_train, column_with_nans, mean_hp)
    X_mean_val, _ = xy(df_val, column_with_nans, mean_hp)

    w0_zero, w_zero = train_linear_regression(X_zero_train, y_train)
    y_zero_pred = w0_zero + X_zero_val.dot(w_zero)
    w0_mean, w_mean = train_linear_regression(X_mean_train, y_train)
    y_mean_pred = w0_mean + X_mean_val.dot(w_mean)

    scores = pd.Series(
        {
            "zero": rmse(y_val, y_zero_pred),
            "mean": rmse(y_val, y_mean_pred),
        }
    ).round(3)
    scores["best"] = scores.idxmin()
    scores
    return (
        X_zero_train,
        X_zero_val,
        rmse,
        train_linear_regression,
        xy,
        y_train,
        y_val,
    )


@app.cell
def _(
    X_zero_train, X_zero_val, np, pd, rmse, train_linear_regression, xy, y_train, y_val
):
    # Question 4
    # * Now let's train a regularized linear regression.
    # * For this question, fill the NAs with 0.
    # * Try different values of `r` from this list: `[0, 0.01, 0.1, 1, 5, 10, 100]`.
    # * Use RMSE to evaluate the model on the validation dataset.
    # * Round the RMSE scores to 4 decimal digits. This keeps the small but real
    #   regularization differences visible instead of turning several choices into a tie.
    # * Which `r` gives the best RMSE?
    # If multiple options give the same best RMSE, select the smallest `r`.
    scores_r = pd.DataFrame()
    for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
        w0_r, w_r = train_linear_regression(X_zero_train, y_train, r)
        y_pred_r = w0_r + X_zero_val.dot(w_r)
        scores_r[r] = {
            "rmse": np.round(rmse(y_val, y_pred_r), 4),
            "w0": w0_r.round(4),
            "weights": w_r.round(4),
        }

    scores_r = scores_r.T
    best = (scores_r["rmse"].idxmin(), scores_r["rmse"].min())
    print("best:", best)
    scores_r
    return (train_linear_regression,)


@app.cell
def _(column_with_nans, df, np, rmse, split_shuffled, train_linear_regression, xy):
    # Question 5
    # * We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
    # * Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    # * For each seed, do the train/validation/test split with 60%/20%/20% distribution.
    # * Fill the missing values with 0 and train a model without regularization.
    # * For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
    # * What's the standard deviation of all the scores? To compute the standard deviation, use `np.std`.
    # * Round the result to 3 decimal digits (round(std, 3))
    # What's the value of std?

    seed_scores = []
    for seed in range(10):
        df_train_seed, df_val_seed, _ = split_shuffled(df, seed)
        X_train_seed = df_train_seed.drop(columns="fuel_efficiency_mpg").fillna(
            {column_with_nans: 0}
        )
        y_train_seed = df_train_seed.fuel_efficiency_mpg
        w0_seed, w_seed = train_linear_regression(X_train_seed, y_train_seed)
        X_val_seed = df_val_seed.drop(columns="fuel_efficiency_mpg").fillna(
            {column_with_nans: 0}
        )
        y_val_seed = df_val_seed.fuel_efficiency_mpg
        y_pred_seed = w0_seed + X_val_seed.dot(w_seed)
        seed_scores.append(rmse(y_val_seed, y_pred_seed))

    round(float(np.std(seed_scores)), 3)
    return


@app.cell
def _(
    column_with_nans,
    df,
    pd,
    rmse,
    split_shuffled,
    train_linear_regression,
    xy,
):
    # Question 6
    # * Split the dataset like previously, use seed 9.
    # * Combine train and validation datasets.
    # * Fill the missing values with 0 and train a model with `r=0.001`.
    # * What's the RMSE on the test dataset?

    df_s9_train, df_s9_val, df_s9_test = split_shuffled(df, seed=9)
    df_s9_full = pd.concat([df_s9_train, df_s9_val])
    X_s9_full, y_s9_full = xy(df_s9_full, column_with_nans, 0)
    w0_full, w_full = train_linear_regression(X_s9_full, y_s9_full, r=0.001)
    X_test, y_test = xy(df_s9_test, column_with_nans, 0)
    y_pred = w0_full + X_test.dot(w_full)
    round(rmse(y_test, y_pred), 3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
