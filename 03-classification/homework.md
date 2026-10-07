## Homework 3

> [!NOTE]
> This homework uses the pinned 2026 lead-scoring release in the course
> repository. The plan and report are available in `cohorts/2026/data/`.

### Dataset

In this homework, we will use the 2026 lead-scoring dataset. Download it from [here](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv).

Or you can do it with `wget`:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv
```

Loaded with `pd.read_csv("03-classification/course_lead_scoring_2026.csv")`. Working notebook: `03-classification/hw03.py` (marimo).

In this dataset our desired target for classification task will be `converted` variable - has the client signed up to the platform or not.

### Data preparation

* Check if the missing values are presented in the features.
* If there are missing values:
    * For categorical features, replace them with 'NA'
    * For numerical features, replace with with 0.0 

### Solution

```python
categorical = list(df.select_dtypes(include="str").columns)
# converted is the target, so no filling
numerical = list(df.select_dtypes(exclude="str").columns.drop("converted"))

df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)
```

**Result**: `lead_source`, `industry`, `employment_status` and `location` had missing values; none of the numerical columns did.

### Question 1

What is the most frequent observation (mode) for the column `industry`?

- `NA`
- `technology`
- `healthcare`
- `retail`

### Solution

```python
df["industry"].mode()
```

**Result**:

```
['technology']
```

**Answer**: `technology`

### Question 2

Create the [correlation matrix](https://www.google.com/search?q=correlation+matrix) for the numerical features of your dataset.
In a correlation matrix, you compute the correlation coefficient between every pair of features.

What are the two features that have the biggest correlation?

- `interaction_count` and `lead_score`
- `number_of_courses_viewed` and `lead_score`
- `number_of_courses_viewed` and `interaction_count`
- `annual_income` and `interaction_count`

Only consider the pairs above when answering this question.

### Solution

```python
corr = df[numerical].corr()

pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .dropna()
    .sort_values(ascending=False)
)
```

**Result**:

```
interaction_count          lead_score    0.9157
number_of_courses_viewed   lead_score    0.7572
number_of_courses_viewed   interaction_count   0.7216
annual_income              lead_score    0.2295
annual_income              number_of_courses_viewed    0.1613
annual_income              interaction_count   0.1228
```

**Answer**: `interaction_count` and `lead_score` (0.9157)

### Split the data

- Split the data with these exact calls:

```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)
```

- Make sure that the target value `converted` is not in your dataframe.

Applied exactly: `converted` stays out of the feature matrices (the `xy`-style extraction separates it from each split).

### Question 3

- Calculate the mutual information score between `converted` and other categorical variables in the dataset. Use the training set only.
- Round the scores to 2 decimals using `round(score, 2)`.

Which of these variables has the biggest mutual information score?

- `industry`
- `location`
- `lead_source`
- `employment_status`

### Solution

```python
mi = {c: round(mutual_info_score(df_train[c], df_train.converted), 2) for c in categorical}
```

**Result**:

```
{'lead_source': 0.03, 'employment_status': 0.02, 'industry': 0.0, 'location': 0.0}
```

**Answer**: `lead_source` (0.03)

### Question 4

- Now let's train a logistic regression.
- Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
- Fit the model on the training dataset.
  - To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
  - `model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
- Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

What accuracy did you get?

- 0.55
- 0.65
- 0.75
- 0.85

### Solution

```python
dv = DictVectorizer(sparse=False)
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train_ohe = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val_ohe = dv.transform(val_dicts)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train_ohe, y_train)

accuracy = (model.predict(X_val_ohe) == y_val).mean()
```

**Result**:

```
0.645
```

**Answer**: `0.65`

### Question 5

- Let's find the least useful feature using the _feature elimination_ technique.
- Train a model using the same features and parameters as in Q4 (without rounding).
- Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
- For each feature, calculate the difference between the original accuracy and the accuracy without the feature.

Which of following feature has the smallest difference?

- `'lead_source'`
- `'number_of_courses_viewed'`
- `'interaction_count'`

> **Note**: The difference doesn't have to be positive.

### Solution

```python
baseline = (model.predict(X_val_ohe) == y_val).mean()

elimination = {}
for f in ["lead_source", "number_of_courses_viewed", "interaction_count"]:
    cols = [c for c in categorical + numerical if c != f]

    dv_f = DictVectorizer(sparse=False)
    X_tr_f = dv_f.fit_transform(df_train[cols].to_dict(orient="records"))
    X_val_f = dv_f.transform(df_val[cols].to_dict(orient="records"))

    m = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    m.fit(X_tr_f, y_train)
    elimination[f] = round(baseline - float((m.predict(X_val_f) == y_val).mean()), 6)
```

**Result**:

```
{'lead_source': 0.003, 'number_of_courses_viewed': 0.002, 'interaction_count': 0.044}
```

**Answer**: `number_of_courses_viewed` (smallest difference: 0.002)

### Question 6

- Now let's train a regularized logistic regression.
- Let's try the following values of the parameter `C`: `[0.000001, 0.00001, 0.0001, 0.001]`.
- Train models using all the features as in Q4.
- Calculate the accuracy on the validation dataset and round it to 3 decimal digits.

Which of these `C` leads to the best accuracy on the validation set?

- 0.000001
- 0.00001
- 0.0001
- 0.001

> **Note**: If there are multiple options, select the smallest `C`.

### Solution

```python
c_scores = {}
for c in [0.000001, 0.00001, 0.0001, 0.001]:
    m = LogisticRegression(solver="liblinear", C=c, max_iter=1000, random_state=42)
    m.fit(X_train_ohe, y_train)
    c_scores[c] = round(float((m.predict(X_val_ohe) == y_val).mean()), 3)
```

**Result**:

```
{1e-06: 0.598, 1e-05: 0.598, 0.0001: 0.613, 0.001: 0.645}
```

**Answer**: `0.001` (best accuracy: 0.645)

## Submit the results

- Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03
- The numerical options are calculated from the pinned 2026 release. Use the value that matches your calculation; do not choose a merely close value.
