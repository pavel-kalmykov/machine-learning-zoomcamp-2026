## Homework 4

> [!NOTE]
> This homework uses the pinned 2026 lead-scoring release in the course
> repository. The plan and report are available in `cohorts/2026/data/`.

In this homework, we will use the 2026 lead-scoring dataset. Download it from [here](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv).

Loaded with `pd.read_csv("03-classification/course_lead_scoring_2026.csv")`. Working notebook: `04-evaluation/hw04.py` (marimo).

In this dataset our desired target for classification task will be `converted` variable - has the client signed up to the platform or not. 

### Data preparation

* Check if the missing values are presented in the features.
* If there are missing values:
    * For categorical features, replace them with 'NA'
    * For numerical features, replace with with 0.0 

### Solution

```python
categorical = list(df.select_dtypes(include="str").columns)
numerical = list(df.select_dtypes(exclude="str").columns.drop("converted"))

df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)
```

`converted` has no NaNs (and stays out of the dataframe before the split, as required below).

### Question 1: ROC AUC feature importance

ROC AUC could also be used to evaluate feature importance of numerical variables. 

Let's do that

* For each numerical variable, use it as score (aka prediction) and compute the AUC with the `y` variable as ground truth.
* Use the training dataset for that

If your AUC is < 0.5, invert this variable by putting "-" in front

(e.g. `-df_train['balance']`)

AUC can go below 0.5 if the variable is negatively correlated with the target variable. You can change the direction of the correlation by negating this variable - then negative correlation becomes positive.

Which numerical variable (among the following 4) has the highest AUC?

- `lead_score`
- `number_of_courses_viewed`
- `interaction_count`
- `annual_income`

### Solution

```python
for feat in numerical:
    auc = roc_auc_score(df_train["converted"], df_train[feat])
    if auc < 0.5:
        auc = roc_auc_score(df_train["converted"], -df_train[feat])
    print(feat, round(auc, 3))
```

**Result**:

```
lead_score 0.788
number_of_courses_viewed 0.723
interaction_count 0.765
annual_income 0.608
```

**Answer**: `lead_score` (AUC 0.788)

### Question 2: Training the model

Apply one-hot-encoding using `DictVectorizer` and train the logistic regression with these parameters:

```python
LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
```

What's the AUC of this model on the validation dataset? (round to 3 digits)

- 0.532
- 0.632
- 0.732
- 0.832

### Solution

```python
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train_ohe = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val_ohe = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_ohe, df_train["converted"])

y_pred = model.predict_proba(X_val_ohe)[:, 1]
roc_auc_score(df_val["converted"], y_pred)
```

**Result**:

```
0.732
```

**Answer**: `0.732`

### Question 3: Precision and Recall

Now let's compute precision and recall for our model.

* Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
* For each threshold, compute precision and recall
* Plot them

At which threshold precision and recall curves intersect? Ignore thresholds at
which both metrics are zero, and choose the threshold with the smallest
absolute difference between precision and recall.

* 0.43
* 0.63
* 0.73
* 0.83

### Solution

```python
thresholds = np.linspace(0.0, 1.0, 101)
scores = []
for t in thresholds:
    pred = (y_pred >= t).astype(int)
    tp = ((pred == 1) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    r = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    scores.append((t, p, r))

valid = [(t, p, r) for t, p, r in scores if not (np.isnan(p) or np.isnan(r))]
crossing = min(valid, key=lambda x: abs(x[1] - x[2]))
```

**Result**:

```
t = 0.63 (precision 0.733, recall 0.725)
```

**Answer**: `0.63`

### Question 4: F1 score

Precision and recall are conflicting - when one grows, the other goes down. That's why they are often combined into the F1 score - a metrics that takes into account both

This is the formula for computing F1:

$$F_1 = 2 \cdot \cfrac{P \cdot R}{P + R}$$

Where $P$ is precision and $R$ is recall.

Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01

At which threshold F1 is maximal?

- 0.21
- 0.41
- 0.61
- 0.81

### Solution

```python
f1s = [(t, 2 * p * r / (p + r)) for t, p, r in valid]
best_f1 = max(f1s, key=lambda x: x[1])
```

**Result**:

```
t = 0.41, F1 = 0.758
```

**Answer**: `0.41`

### Question 5: 5-Fold CV

Use the `KFold` class from Scikit-Learn to evaluate our model on 5 different folds:

```
KFold(n_splits=5, shuffle=True, random_state=1)
```

* Iterate over different folds of `df_full_train`
* Split the data into train and validation
* Train the model on train with these parameters: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`
* Use AUC to evaluate the model on validation

How large is standard deviation of the scores across different folds?

- 0.001
- 0.007
- 0.013
- 0.060

### Solution

```python
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
fold_aucs = []
for tr_idx, v_idx in kfold.split(df_full_train):
    dtr = df_full_train.iloc[tr_idx]
    dv_f = DictVectorizer(sparse=False)
    X_tr = dv_f.fit_transform(dtr[categorical + numerical].to_dict(orient="records"))
    y_tr = dtr.converted.values
    X_v = dv_f.transform(df_full_train.iloc[v_idx][categorical + numerical].to_dict(orient="records"))
    y_v = df_full_train.iloc[v_idx].converted.values
    m = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    m.fit(X_tr, y_tr)
    fold_aucs.append(roc_auc_score(y_v, m.predict_proba(X_v)[:, 1]))
round(float(np.std(fold_aucs)), 3)
```

**Result**:

```
0.007
```

**Answer**: `0.007` (low std: stable model)

### Question 6: Hyperparameter Tuning

Now let's use 5-Fold cross-validation to find the best parameter `C`

* Iterate over the following `C` values: `[0.000001, 0.001, 1]`
* Initialize `KFold` with the same parameters as previously
* Use these parameters for the model: `LogisticRegression(solver='liblinear', C=C, max_iter=1000)`
* Compute the mean ROC AUC as well as the std (round the mean and std to 3
  decimal digits).

Which `C` leads to the best mean ROC AUC?

- 0.000001
- 0.001
- 1

If you have ties, select the score with the lowest std. If you still have ties, select the smallest `C`.

### Solution

```python
for C in [0.000001, 0.001, 1]:
    aucs_c = []
    for tr_idx, v_idx in kfold.split(df_full_train):
        dtr = df_full_train.iloc[tr_idx]
        dv_f = DictVectorizer(sparse=False)
        X_tr = dv_f.fit_transform(dtr[categorical + numerical].to_dict(orient="records"))
        y_tr = dtr.converted.values
        X_v = dv_f.transform(df_full_train.iloc[v_idx][categorical + numerical].to_dict(orient="records"))
        y_v = df_full_train.iloc[v_idx].converted.values
        m = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
        m.fit(X_tr, y_tr)
        aucs_c.append(roc_auc_score(y_v, m.predict_proba(X_v)[:, 1]))
    print(C, round(float(np.mean(aucs_c)), 3), round(float(np.std(aucs_c)), 3))
```

**Result**:

```
C=0.000001: mean 0.617, std 0.019
C=0.001:    mean 0.749, std 0.010
C=1:        mean 0.732, std 0.007
```

**Answer**: `0.001` (best mean AUC: 0.749)

## Submit the results

- Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04
- The numerical options are calculated from the pinned 2026 release. Use the value that matches your calculation; do not choose a merely close value.
