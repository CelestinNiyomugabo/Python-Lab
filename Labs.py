# Introduction to Numerical Python
import numpy as np

x = np.array ([3, 4, 5])
y = np.array ([4, 9, 7])
x + y

x = np.array ([[1 , 2], [3, 4]])
xx = np.array ([[1 , 2], [3, 4]])
x

x.ndim
x.dtype
np.array ([[1 , 2], [3.0 , 4]]).dtype

np.array ([[1 , 2], [3, 4]], float).dtype
x.shape

# Sum 
x = np.array ([1, 2, 3, 4])
x.sum ()
np.sum(x)

# Reshape function
x = np.array ([1, 2, 3, 4, 5, 6])
print('beginning x:\n', x)
x_reshape = x.reshape ((2, 3))
print('reshaped x:\n', x_reshape )
x = np.array ([1, 2, 3, 4, 5, 6])
print('beginning x:\n', x)
x_reshape = x.reshape ((2, 3))
print('reshaped x:\n', x_reshape )

# Square root





# Lists
x = [3, 4, 5]
y = [4, 9, 7]
print(x + y)  # Concatenation

# Numerical Python
import numpy as np
a = np.array([1, 3, 4])
b = np.array([2, 1, 2])
print(a + b)  # Element-wise addition

# Graphics
import matplotlib.pyplot as plt
x = np.linspace(-2, 2, 100)
y = x**2
plt.plot(x, y)
plt.title("y = x^2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()



# Chapter 3: Linear Regression
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

# Load the Boston dataset
Boston = load_data("Boston")

# Simple linear regression: medv vs. lstat
y = Boston["medv"]
X = sm.add_constant(Boston["lstat"])
model = sm.OLS(y, X).fit()

# Summarize results
print(model.summary())

# Plot with regression line
ax = Boston.plot.scatter("lstat", "medv")
xlim = ax.get_xlim()
ylim = [model.params[1] * x + model.params[0] for x in xlim]
ax.plot(xlim, ylim, 'r--', linewidth=2)

