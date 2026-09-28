import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # Q1. Pandas version
    # What version of Pandas did you install?
    import pandas as pd

    pd.__version__
    return (pd,)


@app.cell
def _(pd):
    # Q2. Records count
    # How many records are in the dataset?
    df = pd.read_csv("01-intro/car_fuel_efficiency_2026.csv")
    df.shape  # (rows, cols)
    return (df,)


@app.cell
def _(df):
    df.head()  # For reference
    return


@app.cell
def _(df):
    # Q3. Fuel types
    # How many fuel types are presented in the dataset?
    df["fuel_type"].nunique()
    return


@app.cell
def _(df):
    # Q4. Missing values
    # How many columns in the dataset have missing values?
    df.isna().any()  # .sum()
    return


@app.cell
def _(df):
    # Q5. Max fuel efficiency
    # What’s the maximum fuel efficiency of cars from Asia?
    df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max()
    return


@app.cell
def _(df):
    # Q6. Median value of horsepower
    # 1. Find the median value of the horsepower column in the dataset.
    m_before = df["horsepower"].median()
    m_before
    return (m_before,)


@app.cell
def _(df):
    # 2. Next, calculate the most frequent value of the same horsepower column.
    hp_mode = df["horsepower"].mode()
    hp_mode  # Mode is a Series because there can be multiple modes (draw)
    return (hp_mode,)


@app.cell
def _(df, hp_mode):
    # 3. Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
    df_hp_without_nan = df["horsepower"].fillna(hp_mode[0])
    df_hp_without_nan.isna().any()
    return (df_hp_without_nan,)


@app.cell
def _(df_hp_without_nan):
    # 4. Now, calculate the median value of horsepower once again.
    m_after = df_hp_without_nan.median()
    m_after
    return (m_after,)


@app.cell
def _(m_after, m_before, pd):
    # Has it changed?
    pd.DataFrame(
        {"value": [m_before, m_after], "delta": [0.0, m_after - m_before]},
        index=["median before", "median after"],
    )
    return


@app.cell
def _(df):
    # Q7. Sum of weights
    # 1. Select all the cars from Asia
    df_asia = df[df["origin"] == "Asia"]
    df_asia
    return (df_asia,)


@app.cell
def _(df_asia):
    # 2. Select only columns `vehicle_weight` and `model_year`
    df_asia_select = df_asia[["vehicle_weight", "model_year"]]
    df_asia_select
    return (df_asia_select,)


@app.cell
def _(df_asia_select):
    # 3. Select the first 7 values
    df_asia_select_7 = df_asia_select[:7]
    df_asia_select_7
    return (df_asia_select_7,)


@app.cell
def _(df_asia_select_7):
    # 4. Get the underlying NumPy array. Let's call it `X`.
    X = df_asia_select_7.to_numpy()
    X
    return (X,)


@app.cell
def _(X):
    # 5. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
    XTX = X.T @ X
    XTX
    return (XTX,)


@app.cell
def _(XTX):
    # 6. Invert `XTX`.
    import numpy as np

    XTX_inv = np.linalg.inv(XTX)
    XTX_inv
    return XTX_inv, np


@app.cell
def _(np):
    # 7. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100, 1200]`.
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    y
    return (y,)


@app.cell
def _(X, XTX_inv, y):
    # 8. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
    w = XTX_inv @ X.T @ y
    w
    return (w,)


@app.cell
def _(w):
    # 9. What's the sum of all the elements of the result?
    w.sum()
    return


if __name__ == "__main__":
    app.run()
