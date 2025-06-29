# Numpy
import numpy as np

x = np.array([2,4,5,6,2,6,1,5])
x

y = x+2
y

y2 = np.array([[12,34,23,43], [23,1,12,23]])
y2.ndim
y2.shape

np.sum(y2)

y3 = y.reshape(2,4)

rand1 = np.random.default_rng(1)
y4 = rand1.normal(size=100)
y4

y5 = rand1.standard_normal((30,2))
y5

np.corrcoef(y2[0], y2[1])

# Ploting graphs
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (10,6),
                       ncols=2,
                       nrows=1)
ax[0].plot(x)
ax[1].plot(y2[0], y2[1], 'o')


fig , ax = plt.subplots (figsize =(8, 8))
x = np.linspace (-np.pi , np.pi , 50)
y = x
f = np.multiply .outer(np.cos(y), 1 / (1 + x**2))
ax.contour(x, y, f)


# Linear regression
#===================================================================
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

import statsmodels.api as sm
from statsmodels.stats. outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import ( ModelSpec as MS ,
summarize ,
poly)


Boston = load_data ("Boston")
Boston.columns

model = sm.OLS(Boston['medv'],
               sm.add_constant(Boston[['lstat', 'age']])).fit()
print(model.summary())


Boston.head()

