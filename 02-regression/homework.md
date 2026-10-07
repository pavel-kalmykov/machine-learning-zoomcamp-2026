## Homework 2

> [!NOTE]
> This homework uses the pinned 2026 car fuel-efficiency release in the course
> repository. The plan and report are available in `cohorts/2026/data/`.

### Dataset

For this homework, we'll use the 2026 Car Fuel Efficiency dataset. Download it from <a href='https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv'>here</a>.

You can do it with wget:
```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

The goal of this homework is to create a regression model for predicting the car fuel efficiency (column `'fuel_efficiency_mpg'`).

Loaded with `df = pd.read_csv("01-intro/car_fuel_efficiency_2026.csv")`, filtered to the five listed columns. Working notebook: `02-regression/hw02.py` (marimo).

### Preparing the dataset 

Use only the following columns:

* `'engine_displacement'`,
* `'horsepower'`,
* `'vehicle_weight'`,
* `'model_year'`,
* `'fuel_efficiency_mpg'`

### EDA

* Look at the `fuel_efficiency_mpg` variable. Does it have a long tail? 

**No.** Skewness is 0.08 (well inside the ±0.5 symmetric range) and mean ≈ median, so no log transform is needed.

### Question 1

There's one column with missing values. What is it?

* `'engine_displacement'`
* `'horsepower'`
* `'vehicle_weight'`
* `'model_year'`

### Solution

```python
df.columns[df.isnull().any()][0]
```

**Result**:

```
horsepower
```

**Answer**: `'horsepower'` (877 NaNs)

## Question 2

What's the median (50% percentile) for variable `'horsepower'`?

- 204
- 254
- 304
- 354

### Solution

```python
df["horsepower"].median()
```

**Result**:

```
254.0
```

**Answer**: `254` (pandas skips the NaNs)

## Question 3

* We need to deal with missing values for the column from Q1.
* We have two options: fill it with 0 or with the mean of this variable.
* Try both options. For each, train a linear regression model without regularization using the code from the lessons.
* For computing the mean, use the training only!
* Use the validation dataset to evaluate the models and compare the RMSE of each option.
* Round the RMSE scores to 3 decimal digits using `round(score, 3)`. This
  keeps the imputation difference visible in this release.
* Which option gives better RMSE?

Options:

- With 0
- With mean
- Both are equally good

### Solution

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


X_train, y_train = df_train.drop(columns="fuel_efficiency_mpg"), df_train.fuel_efficiency_mpg
X_val, y_val = df_val.drop(columns="fuel_efficiency_mpg"), df_val.fuel_efficiency_mpg

mean_hp = X_train[column_with_nans].mean()   # training only
X_zero_train = X_train.fillna({column_with_nans: 0})
X_zero_val = X_val.fillna({column_with_nans: 0})
X_mean_train = X_train.fillna({column_with_nans: mean_hp})
X_mean_val = X_val.fillna({column_with_nans: mean_hp})

w0_zero, w_zero = train_linear_regression(X_zero_train, y_train)
y_zero_pred = w0_zero + X_zero_val.dot(w_zero)
w0_mean, w_mean = train_linear_regression(X_mean_train, y_train)
y_mean_pred = w0_mean + X_mean_val.dot(w_mean)

scores = pd.Series({
    "zero": rmse(y_val, y_zero_pred),
    "mean": rmse(y_val, y_mean_pred),
}).round(3)
scores["best"] = scores.idxmin()
```

**Result**:

```
zero    2.205
mean    2.202
best:     mean
```

**Answer**: With mean (2.202 vs 2.205). The validation NaNs are also filled with the training mean.

## Question 4

* Now let's train a regularized linear regression.
* For this question, fill the NAs with 0. 
* Try different values of `r` from this list: `[0, 0.01, 0.1, 1, 5, 10, 100]`.
* Use RMSE to evaluate the model on the validation dataset.
* Round the RMSE scores to 4 decimal digits. This keeps the small but real
  regularization differences visible instead of turning several choices into a tie.
* Which `r` gives the best RMSE?

If multiple options give the same best RMSE, select the smallest `r`.

Options:

- 0
- 0.01
- 0.1
- 1
- 5
- 10
- 100

### Solution

```python
scores_r = pd.DataFrame()
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0_zero_r, w_zero_r = train_linear_regression_reg(X_zero_train, y_train, r)
    y_zero_pred_r = w0_zero_r + X_zero_val.dot(w_zero_r)
    scores_r[r] = {
        "rmse": np.round(rmse(y_val, y_zero_pred_r), 4),
        "w0": w0_zero_r.round(4),
        "weights": w_zero_r.round(4),
    }

scores_r = scores_r.T
best = (scores_r["rmse"].idxmin(), scores_r["rmse"].min())
```

**Result**:

```
r=0:    2.2053
r=0.01: 2.2058
r=0.1:  2.2241
r=1:    2.3492
r=5:    2.4094
r=10:   2.4195
r=100:  2.4292
best:   (0, 2.2053)
```

**Answer**: `0` (best RMSE 2.2053; with this dataset regularization does not improve validation RMSE, so the tie rule selects the smallest r)

## Question 5 

* We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
* Try different seed values: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
* For each seed, do the train/validation/test split with 60%/20%/20% distribution.
* Fill the missing values with 0 and train a model without regularization.
* For each seed, evaluate the model on the validation dataset and collect the RMSE scores. 
* What's the standard deviation of all the scores? To compute the standard deviation, use `np.std`.
* Round the result to 3 decimal digits (`round(std, 3)`)

What's the value of std?

- 0.006
- 0.016
- 0.029
- 0.036

> Note: Standard deviation shows how different the values are.
> If it's low, then all values are approximately the same.
> If it's high, the values are different. 
> If standard deviation of scores is low, then our model is *stable*.

### Solution

```python
seed_scores = []
for seed in range(10):
    df_train_seed, df_val_seed, _ = split_shuffled(df, seed)
    X_train = df_train_seed.drop(columns="fuel_efficiency_mpg").fillna(
        {column_with_nans: 0}
    )
    y_train = df_train_seed.fuel_efficiency_mpg
    w0, w = train_linear_regression(X_train, y_train)
    X_val = df_val_seed.drop(columns="fuel_efficiency_mpg").fillna(
        {column_with_nans: 0}
    )
    y_val = df_val_seed.fuel_efficiency_mpg
    y_pred = w0 + X_val.dot(w)
    seed_scores.append(rmse(y_val, y_pred))

round(float(np.std(seed_scores)), 3)
```

**Result**:

```
0.029
```

**Answer**: `0.029` (low std: the model is stable across seeds)

## Question 6

* Split the dataset like previously, use seed 9.
* Combine train and validation datasets.
* Fill the missing values with 0 and train a model with `r=0.001`. 
* What's the RMSE on the test dataset?

Options:

- 0.236
- 2.236
- 22.10
- 221.0

### Solution

```python
df_s9_train, df_s9_val, df_s9_test = split_shuffled(df, seed=9)
df_s9_full = pd.concat([df_s9_train, df_s9_val])
X_s9 = df_s9_full.drop(columns="fuel_efficiency_mpg").fillna({column_with_nans: 0})
y_s9 = df_s9_full.fuel_efficiency_mpg
w0, w = train_linear_regression_reg(X_s9, y_s9, r=0.001)
X_test = df_s9_test.drop(columns="fuel_efficiency_mpg").fillna({column_with_nans: 0})
y_test = df_s9_test.fuel_efficiency_mpg
y_pred = w0 + X_test.dot(w)
round(rmse(y_test, y_pred), 3)
```

**Result**:

```
2.236
```

**Answer**: `2.236`

## Submit the results

* Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02
* The numerical options are calculated from the pinned 2026 release. Use the value that matches your calculation; do not choose a merely close value.
