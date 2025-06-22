import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, summarize, poly

# Load Boston housing data
Boston = load_data("Boston")
print(Boston.columns)

# Simple linear regression with lstat predicting medv
X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]), 'lstat': Boston['lstat']})
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))

# Using ModelSpec for transformations
design = MS(['lstat'])
X = design.fit_transform(Boston)
print(X[:4])

# Diagnostic plots and predictions
new_df = pd.DataFrame({'lstat':[5, 10, 15]})
newX = design.transform(new_df)
new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print(new_predictions.conf_int(alpha=0.05))

# Plotting function
def abline(ax, b, m, *args, **kwargs):
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

# Scatter plot with regression line
ax = Boston.plot.scatter('lstat', 'medv')
abline(ax, results.params[0], results.params[1], color='red', linestyle='--', linewidth=2)

# Diagnostic plots
fitted_values = results.fittedvalues
residuals = results.resid
influence = results.get_influence()
studentized_residuals = influence.resid_studentized_external
leverage = influence.hat_matrix_diag

# Plot fitted vs residuals
_, ax = subplots(figsize=(8, 6))
ax.scatter(fitted_values, residuals, facecolors='none', edgecolors='b')
ax.set_xlabel('Fitted values')
ax.set_ylabel('Residuals')

# Plot studentized residuals vs leverage
_, ax = subplots(figsize=(8, 6))
ax.scatter(leverage, studentized_residuals, facecolors='none', edgecolors='b')
ax.set_xlabel('Leverage')
ax.set_ylabel('Studentized residuals')