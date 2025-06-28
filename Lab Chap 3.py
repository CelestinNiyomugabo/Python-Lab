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


# Multiple linear regression
X = MS(['lstat', 'age']).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)


terms = Boston.columns.drop('medv')
terms

# Multiple linear regression with all predictors
X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

minus_age = Boston.columns.drop(['medv', 'age']) 
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
summarize(model1.fit())


# Multivariate Goodness of Fit
vals = [VIF(X, i)
        for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif':vals},
                   index=X.columns[1:])
vif



# Interaction terms
X = MS(['lstat',
        'age',
        ('lstat', 'age')]).fit_transform(Boston)
model2 = sm.OLS(y, X)
summarize(model2.fit())

# Non-linear Transformations of the Predictors
X = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
summarize(results3)


anova_lm(results1, results3)

# Residuals vs Fitted values plot
ax = subplots(figsize=(8,8))[1]
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')


# Qualitative Predictors
#===========================================
Carseats = load_data('Carseats')
Carseats.columns

allvars = list(Carseats.columns.drop('Sales'))
y = Carseats['Sales']
final = allvars + [('Income', 'Advertising'),
                   ('Price', 'Age')]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
summarize(model.fit())