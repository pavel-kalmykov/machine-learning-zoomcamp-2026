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

Loaded with `pd.read_csv("01-intro/car_fuel_efficiency_2026.csv")`, filtered to the five listed columns. Working notebook: `02-regression/hw02.py` (marimo).

### Preparing the dataset 

Use only the following columns:

* `'engine_displacement'`,
* `'horsepower'`,
* `'vehicle_weight'`,
* `'model_year'`,
* `'fuel_efficiency_mpg'`

### EDA

* Look at the `fuel_efficiency_mpg` variable. Does it have a long tail? 

**No**: skewness is 0.08 (well inside the ±0.5 symmetric range) and mean ≈ median, so the distribution is roughly bell-shaped and no log transform is needed.

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
horsepower   877
```

**Answer**: `'horsepower'`

### Question 2

What's the median (50% percentile) for variable `'horsepower'`?

- 204
- 254
- 304
- 354

### Solution

```python
df["horsepower"].median()   # pandas skips the NaNs
```

**Result**:

```
254.0
```

**Answer**: `254`

### Prepare and split the dataset

Shuffle the filtered dataset and create the split exactly as in the lecture:

```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]
```

For Q5, repeat the same block with each listed seed. For Q6, use seed `9`.

Implemented as `split_shuffled(df, seed)` in the notebook; seed 42 used below.

### Question 3

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
mean_hp = df_train["horsepower"].mean()          # training only
X_zero_train = df_train.fillna({"horsepower": 0})
X_zero_val = df_val.fillna({"horsepower": 0})
X_mean_train = df_train.fillna({"horsepower": mean_hp})
X_mean_val = df_val.fillna({"horsepower": mean_hp})

w = train_linear_regression(X_zero_train, y_train)
rmse(y_val, w[0] + X_zero_val.dot(w[1:]))

w = train_linear_regression(X_mean_train, y_train)
rmse(y_val, w[0] + X_mean_val.dot(w[1:]))
```

**Result**:

```
RMSE with 0:    2.205
RMSE with mean: 2.202
```

**Answer**: With mean (2.202 vs 2.205). The validation NaNs are also filled with the training mean.

### Question 4

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
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w = train_linear_regression(X_zero_train, y_train, r)
    scores_r[r] = round(rmse(y_val, w[0] + X_zero_val.dot(w[1:])), 4)
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
```

**Answer**: `0` (the smallest r, per the tie rule: with this dataset the regularization does not improve validation RMSE).

### Question 5 

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
    df_train, df_val, _ = split_shuffled(df, seed)
    w = train_linear_regression(X_train_zero_filled, y_train)
    seed_scores.append(rmse(y_val, y_pred))
round(float(np.std(seed_scores)), 3)
```

**Result**:

```
0.029
```

**Answer**: `0.029` (a low std: the model is stable across seeds).

### Question 6

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
df_train9, df_val9, df_test9 = split_shuffled(df, seed=9)
df_full = pd.concat([df_train9, df_val9])
w = train_linear_regression(X_full, y_full, r=0.001)
rmse(y_test, y_pred)
```

**Result**:

```
2.236
```

**Answer**: `2.236`

## Submit the results

* Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02
* The numerical options are calculated from the pinned 2026 release. Use the value that matches your calculation; do not choose a merely close value.
