import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("03-classification/course_lead_scoring_2026.csv")
    return (df,)


@app.cell
def _(df):
    # Data preparation
    # Check if the missing values are presented in the features.
    df.isna().sum()
    return


@app.cell
def _(df):
    # If there are missing values:
    # * For categorical features, replace them with 'NA'
    # * For numerical features, replace with with 0.0
    categorical = list(df.select_dtypes(include="str").columns)
    # converted is the target, so no filling
    numerical = list(df.select_dtypes(exclude="str").columns.drop("converted"))

    df[categorical] = df[categorical].fillna("NA")
    df[numerical] = df[numerical].fillna(0.0)

    categorical, numerical
    return categorical, numerical


@app.cell
def _(df):
    # Split the data into 3 parts with these exact calls:
    from sklearn.model_selection import train_test_split

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_train = df_train["converted"]
    y_val = df_val["converted"]
    return df_full_train, df_train, df_val, y_train, y_val


@app.cell
def _(df_train, numerical, pd, y_train):
    # Question 1: ROC AUC feature importance
    # * For each numerical variable, use it as score (aka prediction) and compute the AUC with the `y` variable as ground truth.
    # * Use the training dataset for that
    # If your AUC is < 0.5, invert this variable by putting "-" in front (e.g. `-df_train['balance']`)
    # Which numerical variable (among the following 4) has the highest AUC?
    from sklearn.metrics import roc_auc_score

    def feature_auc(feat):
        auc = roc_auc_score(y_train, df_train[feat])
        return roc_auc_score(y_train, -df_train[feat]) if auc < 0.5 else auc

    aucs = pd.Series({feat: feature_auc(feat) for feat in numerical}).sort_values(
        ascending=False
    )
    aucs
    return (roc_auc_score,)


@app.cell
def _(categorical, df_train, df_val, numerical, roc_auc_score, y_val):
    # Question 2: Training the model
    # Apply one-hot-encoding using `DictVectorizer` and train the logistic regression with these parameters:
    # ```python
    # LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    # ```

    # What's the AUC of this model on the validation dataset? (round to 3 digits)
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    dv = DictVectorizer(sparse=False)

    train_dict = df_train[categorical + numerical].to_dict(orient="records")
    X_train_ohe = dv.fit_transform(train_dict)

    val_dict = df_val[categorical + numerical].to_dict(orient="records")
    X_val_ohe = dv.transform(val_dict)

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    model.fit(X_train_ohe, df_train["converted"])

    y_pred = model.predict_proba(X_val_ohe)[:, 1]

    round(roc_auc_score(y_val, y_pred), 3)
    return DictVectorizer, LogisticRegression, y_pred


@app.cell
def _(np, pd, plt, y_pred, y_val):
    ### Question 3: Precision and Recall
    # Now let's compute precision and recall for our model.
    # * Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
    # * For each threshold, compute precision and recall
    # * Plot them
    scores = {}

    thresholds = np.round(np.linspace(0, 1, 101), 2)

    for t in thresholds:
        actual_pos = y_val == 1
        actual_neg = y_val == 0

        pred_pos = y_pred >= t
        pred_neg = y_pred < t

        tp = (pred_pos & actual_pos).sum()
        tn = (pred_neg & actual_neg).sum()

        fp = (pred_pos & actual_neg).sum()
        fn = (pred_neg & actual_pos).sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        r = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        scores[t] = {"precision": p, "recall": r}

    scores = pd.DataFrame(scores).T.dropna()

    plt.plot(scores.index, scores.precision, label="precision")
    plt.plot(scores.index, scores.recall, label="recall")
    plt.xlabel("threshold")
    plt.legend()

    # At which threshold precision and recall curves intersect? Ignore thresholds at
    # which both metrics are zero, and choose the threshold with the smallest
    # absolute difference between precision and recall.
    intersection = np.abs(scores.precision - scores.recall).idxmin()
    intersection
    row = scores.loc[intersection]
    plt.plot(
        intersection,
        row.precision,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#b00020",
        markeredgewidth=1,
    )
    plt.annotate(
        f"t = {intersection}",
        xy=(intersection, row.precision),
        xytext=(intersection + 0.05, row.precision - 0.05),
    )
    return (scores,)


@app.cell
def _(plt, scores):
    # Question 4: F1 score
    # Precision and recall are conflicting - when one grows, the other goes down. That's why they are often combined into the F1 score - a metrics that takes into account both
    # This is the formula for computing F1:
    # $$F_1 = 2 \cdot \cfrac{P \cdot R}{P + R}$$
    # Where $P$ is precision and $R$ is recall.
    # Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01
    # At which threshold F1 is maximal?
    scores["f1"] = (
        2 * scores.precision * scores.recall / (scores.precision + scores.recall)
    )
    best_f1 = scores.f1.max()
    best_f1_t = scores.f1.idxmax()

    plt.plot(scores.index, scores.precision, label="precision")
    plt.plot(scores.index, scores.recall, label="recall")
    plt.plot(scores.index, scores.f1, "--", label="F1")
    plt.plot(
        best_f1_t,
        best_f1,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#b00020",
        markeredgewidth=1,
    )
    plt.annotate(
        f"max F1: t = {best_f1_t}",
        xy=(best_f1_t, best_f1),
        xytext=(best_f1_t - 0.25, best_f1 - 0.1),
    )
    plt.xlabel("threshold")
    plt.legend()
    return


@app.cell
def _(
    DictVectorizer,
    LogisticRegression,
    categorical,
    df_full_train,
    np,
    numerical,
    roc_auc_score,
):
    # Question 5: 5-Fold CV
    # Use the `KFold` class from Scikit-Learn to evaluate our model on 5 different folds:
    # ```
    # KFold(n_splits=5, shuffle=True, random_state=1)
    # ```
    # * Iterate over different folds of `df_full_train`
    # * Split the data into train and validation
    # * Train the model on train with these parameters: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`
    # * Use AUC to evaluate the model on validation
    # How large is standard deviation of the scores across different folds?
    from sklearn.model_selection import KFold

    def train(data, target, C=1.0):
        dicts = data[categorical + numerical].to_dict(orient="records")

        dv = DictVectorizer(sparse=False)
        X_train = dv.fit_transform(dicts)

        model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
        model.fit(X_train, target)

        return dv, model

    def predict(data, dv, model):
        dicts = data[categorical + numerical].to_dict(orient="records")

        X = dv.transform(dicts)
        y_pred = model.predict_proba(X)[:, 1]

        return y_pred

    scores_kf = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_idx_kf, val_idx_kf in kfold.split(df_full_train):
        df_train_kf = df_full_train.iloc[train_idx_kf]
        df_val_kf = df_full_train.iloc[val_idx_kf]

        y_train_kf = df_train_kf.converted
        y_val_kf = df_val_kf.converted

        dv_kf, model_kf = train(df_train_kf, y_train_kf)
        y_pred_kf = predict(df_val_kf, dv_kf, model_kf)

        auc_kf = roc_auc_score(y_val_kf, y_pred_kf)
        scores_kf.append(auc_kf)

    np.std(scores_kf)
    return kfold, predict, train


@app.cell
def _(df_full_train, kfold, np, pd, predict, roc_auc_score, train):
    # Question 6: Hyperparameter Tuning
    # Now let's use 5-Fold cross-validation to find the best parameter `C`
    # * Iterate over the following `C` values: `[0.000001, 0.001, 1]`
    # * Initialize `KFold` with the same parameters as previously
    # * Use these parameters for the model: `LogisticRegression(solver='liblinear', C=C, max_iter=1000)`
    # * Compute the mean ROC AUC as well as the std (round the mean and std to 3 decimal digits).
    # Which `C` leads to the best mean ROC AUC?
    def _():
        rows = {}
        for C in [0.000001, 0.001, 1]:
            aucs = []
            for train_idx_kf, val_idx_kf in kfold.split(df_full_train):
                df_train_kf = df_full_train.iloc[train_idx_kf]
                df_val_kf = df_full_train.iloc[val_idx_kf]
                y_train_kf = df_train_kf.converted
                y_val_kf = df_val_kf.converted
                dv_kf, model_kf = train(df_train_kf, y_train_kf, C)
                y_pred_kf = predict(df_val_kf, dv_kf, model_kf)
                aucs.append(roc_auc_score(y_val_kf, y_pred_kf))
            rows[C] = {
                "mean_auc": round(float(np.mean(aucs)), 3),
                "std_auc": round(float(np.std(aucs)), 3),
            }

        scores_c = pd.DataFrame(rows).T
        scores_c.index.name = "C"
        return scores_c

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
