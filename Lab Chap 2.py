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



# Ploting the graph 

import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values from -10 to 10
x = np.linspace(-10, 10, 400)
y = sigmoid(x)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'Sigmoid Function $f(x) = \frac{1}{1 + e^{-x}}$', color='blue')
plt.title('Logistic Regression Sigmoid Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.axhline(0.5, color='red', linestyle='--', linewidth=1)
plt.legend()
plt.show()



# Define the function (second function)
def alt_sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

# Generate x values
x = np.linspace(-10, 10, 400)
y = alt_sigmoid(x)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f(x) = \frac{e^x}{1 + e^x}$', color='green')
plt.title('Alternative Form of the Sigmoid Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.axhline(0.5, color='red', linestyle='--', linewidth=1)
plt.legend()
plt.show()



# Random number generation with a fixed seed
rng = np.random.default_rng(1303)
print(rng.normal(scale=5, size=2))
rng2 = np.random.default_rng(1303)
print(rng2.normal(scale=5, size=2)) 

# Random number generation and basic statistics
rng = np.random.default_rng(3)
y = rng.standard_normal(10)
np.mean(y), y.mean()

# Variance calculation
np.var(y), y.var(), np.mean((y - y.mean())**2)

# Generate a random sample of 10 observations with 3 features
X = rng.standard_normal((10, 3))
X

# Calculate the mean of each feature
X.mean(axis=0)
X.mean(0)


# Plotting a scatter plot with random data
fig, ax = subplots(figsize=(8, 8))
x = rng.standard_normal(100)
y = rng.standard_normal(100)
ax.plot(x, y)

# Plotting a scatter plot with specific data points
fig, ax = subplots(figsize=(8, 8))
ax.plot(x, y, 'o')

# Alternative to the above
fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='o')

# Scatter plot with labeled axes and title
fig, ax = subplots(figsize=(8, 8))
ax.scatter(x, y, marker='o')
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y")


# Contour plot of a function
fig, ax = subplots(figsize=(8, 8))
x = np.linspace(-np.pi, np.pi, 50)
y = x
f = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
ax.contour(x, y, f)

# Contour plot with specified levels
fig, ax = subplots(figsize=(8, 8))
ax.contour(x, y, f, levels=45)

# Filled contour plot
fig, ax = subplots(figsize=(8, 8))
ax.imshow(f)


# Sequences and Slice Notation
seq1 = np.linspace(0, 10, 11)
seq1

# Indexing Data
A = np.array(np.arange(16)).reshape((4, 4))
A

# Importing the Auto dataset
import pandas as pd
Auto = pd.read_csv('Auto.csv')
Auto

# Displaying the first few rows of the Auto dataset
Auto_re = Auto.set_index('name')
Auto_re.iloc[[3,4],[0,2,3]]

# Filtering the Auto dataset based on conditions
Auto_re.loc[lambda df: (df['displacement'] < 300)
                       & (df.index.str.contains('ford')
                       | df.index.str.contains('datsun')),
            ['weight', 'origin']
           ]


# For Loops
total = 0
for value in [3,2,19]:
    total += value
print('Total is: {0}'.format(total))


total = 0
for value in [2,3,19]:
    for weight in [3, 2, 1]:
        total += value * weight
print('Total is: {0}'.format(total))


total = 0
for value, weight in zip([2,3,19],
                         [0.2,0.3,0.5]):
    total += weight * value
print('Weighted average is: {0}'.format(total))


# String Formatting
rng = np.random.default_rng(1)
A = rng.standard_normal((127, 5))
M = rng.choice([0, np.nan], p=[0.8,0.2], size=A.shape)
A += M
D = pd.DataFrame(A, columns=['food',
                             'bar',
                             'pickle',
                             'snack',
                             'popcorn'])
D[:3]


# Scatter plot of 'horsepower' vs 'mpg' from the Auto dataset
fig, ax = subplots(figsize=(8, 8))
ax.plot(Auto['horsepower'], Auto['mpg'], 'o')

# Boxplot of 'mpg' by 'cylinders' from the Auto dataset
fig, ax = subplots(figsize=(8, 8))
Auto.boxplot('mpg', by='cylinders', ax=ax)

# Histogram of 'mpg' from the Auto dataset
fig, ax = subplots(figsize=(8, 8))
Auto.hist('mpg', ax=ax)