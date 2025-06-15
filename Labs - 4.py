import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize)

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
     (LinearDiscriminantAnalysis as LDA,
      QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the Smarket dataset
Smarket = load_data('Smarket')
Smarket

Smarket.columns
Smarket.corr(numeric_only=True)

# Plot trends in Volume
Smarket.plot(y='Volume')


# Logistic regression 
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm = sm.GLM(y,
             X,
             family=sm.families.Binomial())
results = glm.fit()
summarize(results)

results.params

# Predictions
probs = results.predict()
probs[:10]

# Create labels based on probabilities
labels = np.array(['Down']*1250)
labels[probs>0.5] = "Up"
labels

# Confusion table
confusion_table(labels, Smarket.Direction)

# Diagonal elements of the confusion table
(507+145)/1250, np.mean(labels == Smarket.Direction)



# Split the data into training and testing sets
train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
Smarket_test.shape

# Logistic regression on training data
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
probs

# Compare predictions with actual values
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

# Create labels based on probabilities for the test set
labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'
confusion_table(labels, L_test)