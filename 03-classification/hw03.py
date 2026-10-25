import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return np, pd


@app.cell
def _(pd):
    df = pd.read_csv("03-classification/course_lead_scoring_2026.csv")
    return (df,)


@app.cell
def _(df):
    # Data preparation
    # * Check if the missing values are presented in the features.
    df.isna().sum()
    return


@app.cell
def _(df):
    # * If there are missing values:
    #     * For categorical features, replace them with 'NA'
    #     * For numerical features, replace with with 0.0 
    categorical = list(df.select_dtypes(include="str").columns)
    # converted is the target, so no filling
    numerical = list(df.select_dtypes(exclude="str").columns.drop("converted"))

    df[categorical] = df[categorical].fillna("NA")
    df[numerical] = df[numerical].fillna(0.0)

    categorical, numerical
    return categorical, numerical


@app.cell
def _(df):
    # Question 1
    # What is the most frequent observation (mode) for the column industry?
    df.industry.mode()
    return


@app.cell
def _(df, numerical):
    # Question 2
    # Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features.
    corr = df[numerical].corr()
    corr
    return (corr,)


@app.cell
def _(corr, np):
    # What are the two features that have the biggest correlation?
    upper_diag = np.triu(np.ones(corr.shape), k=1).astype(bool)
    (
        corr
        .where(upper_diag)
        .stack()
        .dropna()
        .sort_values(ascending=False)
        .round(4)
    )
    return


@app.cell
def _(df):
    # Split the data
    # Split the data with these exact calls:
    from sklearn.model_selection import train_test_split


    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42
    )

    # Make sure that the target value converted is not in your dataframe.
    def xy(dframe, column="converted"):
        return dframe.drop(columns=column), dframe[column]


    X_train, y_train = xy(df_train)
    X_val, y_val = xy(df_val)
    return X_train, X_val, df_train, df_val, y_train, y_val


@app.cell
def _(X_train, categorical, y_train):
    # Question 3
    # * Calculate the mutual information score between converted and other categorical variables in the dataset. Use the training set only.
    # * Round the scores to 2 decimals using round(score, 2).
    # Which of these variables has the biggest mutual information score?
    from sklearn.metrics import mutual_info_score


    def mutual_info_converted_score(series, y):
        return mutual_info_score(series, y)


    mi = X_train[categorical].apply(lambda series: mutual_info_converted_score(series, y_train))
    mi.sort_values(ascending=False).round(2)
    return


@app.cell
def _(categorical, df_train, df_val, numerical):
    # Question 4
    # * Now let's train a logistic regression.
    # * Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
    from sklearn.feature_extraction import DictVectorizer


    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    X_train_ohe = dv.fit_transform(train_dict)

    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val_ohe = dv.transform(val_dict)
    return DictVectorizer, X_train_ohe, X_val_ohe


@app.cell
def _(X_train_ohe, y_train):
    # * Fit the model on the training dataset.
    #   * To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    #   * model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_ohe, y_train)
    return LogisticRegression, model


@app.cell
def _(X_val_ohe, model, np, y_val):
    # * Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
    # What accuracy did you get?
    y_pred = model.predict_proba(X_val_ohe)[:, 1]
    converted_decision = y_pred >= 0.5
    (
        np.round((y_val == converted_decision).mean(), 2),
        np.round(model.score(X_val_ohe, y_val), 2)  # equivalent
    )
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    X_train,
    X_val,
    X_val_ohe,
    categorical,
    model,
    numerical,
    pd,
    y_train,
    y_val,
):
    # Question 5
    # * Let's find the least useful feature using the feature elimination technique.
    # * Train a model using the same features and parameters as in Q4 (without rounding).
    # * Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
    # * For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
    baseline = model.score(X_val_ohe, y_val)
    elimination = pd.Series()
    for f in ["lead_source", "number_of_courses_viewed", "interaction_count"]:
        cols = [c for c in categorical + numerical if c != f]

        dv_f = DictVectorizer(sparse=False)
        X_train_f = dv_f.fit_transform(X_train[cols].to_dict(orient="records"))
        X_val_f = dv_f.transform(X_val[cols].to_dict(orient="records"))

        model_f = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
        model_f.fit(X_train_f, y_train)
        elimination[f] = baseline - model_f.score(X_val_f, y_val)


    elimination.sort_values(ascending=False), elimination.abs().idxmin()
    return


@app.cell
def _(LogisticRegression, X_train_ohe, X_val_ohe, pd, y_train, y_val):
    accuracies = pd.Series()
    for c in [0.000001, 0.00001, 0.0001, 0.001]:
        model_c = LogisticRegression(solver="liblinear", C=c, max_iter=1000, random_state=42)
        model_c.fit(X_train_ohe, y_train)
        accuracies[c] = model_c.score(X_val_ohe, y_val)


    accuracies.sort_values(ascending=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
