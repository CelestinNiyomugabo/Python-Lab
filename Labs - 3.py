# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
Boston = pd.read_csv("Boston.csv", index_col=0)

# Check dimensions
Boston.shape

# Pairwise scatterplots
sns.pairplot(Boston[["crim", "zn", "indus", "nox", "rm", "age", "medv"]])
plt.show()

# Correlation matrix
Boston.corr()

# Simple linear regression
import statsmodels.api as sm

X = Boston["lstat"]
y = Boston["medv"]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

# Plot regression line
plt.scatter(Boston["lstat"], Boston["medv"])
plt.plot(Boston["lstat"], model.predict(X), color='red')
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()

# Multiple linear regression
X = Boston[["lstat", "rm"]]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Full model
X = Boston.drop("medv", axis=1)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Diagnostic plots
import statsmodels.graphics.api as smg

fig = plt.figure(figsize=(12, 8))
smg.plot_regress_exog(model, "lstat", fig=fig)
plt.show()

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = Boston.drop("medv", axis=1)
X = sm.add_constant(X)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

# Added variable plots
fig = smg.plot_partregress_grid(model)
plt.show()

# Influence plot
fig, ax = plt.subplots(figsize=(8,6))
sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
plt.show()

# Custom prediction
new_data = pd.DataFrame({"lstat": [5.0], "rm": [6.0]})
new_data = sm.add_constant(new_data, has_constant='add')
model.predict(new_data)
