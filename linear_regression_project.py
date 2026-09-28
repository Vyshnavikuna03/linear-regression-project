"""
Simple Linear Regression Project
Predict median house value using the California Housing dataset.

Install first:  pip install scikit-learn numpy pandas matplotlib
Run:            python linear_regression_project.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------
# STEP 1: Data collection / preparation
# ------------------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame
TARGET = "MedHouseVal"  # median house value (in $100,000s)

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nCorrelation with target:\n", df.corr()[TARGET].sort_values(ascending=False))

# Quick exploratory plots
df.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# ------------------------------------------------------------------
# STEP 2: Data preprocessing
# ------------------------------------------------------------------
df = df.drop_duplicates()
df = df.fillna(df.median())  # no-op for this dataset, but good practice

X = df.drop(columns=TARGET)
y = df[TARGET]

# Split BEFORE scaling so no test-set information leaks into training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # learn mean/std from train only
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------------
# STEP 3 & 4: Model building and training
# ------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_s, y_train)

coefs = pd.Series(model.coef_, index=X.columns).sort_values()
print("\nIntercept:", round(model.intercept_, 4))
print("Coefficients:\n", coefs)

# ------------------------------------------------------------------
# STEP 5: Model evaluation
# ------------------------------------------------------------------
def evaluate(name, mdl, Xte, yte):
    pred = mdl.predict(Xte)
    mse = mean_squared_error(yte, pred)
    print(f"{name:<10} MSE={mse:.4f}  RMSE={np.sqrt(mse):.4f}  "
          f"MAE={mean_absolute_error(yte, pred):.4f}  R2={r2_score(yte, pred):.4f}")
    return pred

print("\n--- Test set results ---")
y_pred = evaluate("Linear", model, X_test_s, y_test)

# Actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png")
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals")
plt.savefig("residuals.png")
plt.close()

# Cross-validation gives a more reliable score than one split
cv_r2 = cross_val_score(LinearRegression(), scaler.transform(X), y, cv=5, scoring="r2")
print(f"5-fold CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")

# ------------------------------------------------------------------
# STEP 6 (optional): Improvement
# ------------------------------------------------------------------
# a) Regularization
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
lasso = Lasso(alpha=0.001).fit(X_train_s, y_train)
evaluate("Ridge", ridge, X_test_s, y_test)
evaluate("Lasso", lasso, X_test_s, y_test)

# b) Feature engineering example: log-transform skewed columns, add ratios
X_fe = X.copy()
X_fe["RoomsPerOccup"] = X["AveRooms"] / X["AveOccup"]
X_fe["Population"] = np.log1p(X["Population"])
X_fe["AveOccup"] = np.log1p(X["AveOccup"])

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fe, y, test_size=0.2, random_state=42
)
sc2 = StandardScaler()
lr2 = LinearRegression().fit(sc2.fit_transform(Xf_train), yf_train)
evaluate("Linear+FE", lr2, sc2.transform(Xf_test), yf_test)

# ------------------------------------------------------------------
# Bonus: TRUE "simple" regression with one feature, easy to visualise
# ------------------------------------------------------------------
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    df[["MedInc"]], y, test_size=0.2, random_state=42
)
simple = LinearRegression().fit(Xs_train, ys_train)
print(f"\nSingle-feature (MedInc) R2: {simple.score(Xs_test, ys_test):.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(Xs_test, ys_test, alpha=0.2)
xs = np.linspace(Xs_test["MedInc"].min(), Xs_test["MedInc"].max(), 100)
plt.plot(xs, simple.predict(pd.DataFrame({"MedInc": xs})), "r")
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.title("Simple Linear Regression")
plt.savefig("simple_regression.png")
plt.close()

# ------------------------------------------------------------------
# Predict on a new sample
# ------------------------------------------------------------------
sample = X_test.iloc[[0]]
print("\nSample prediction:", model.predict(scaler.transform(sample))[0],
      "| actual:", y_test.iloc[0])