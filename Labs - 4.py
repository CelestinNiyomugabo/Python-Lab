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
# Smarket.plot(y='Volume')



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


# Logistic regression with lagged variables
model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'
confusion_table(labels, L_test)

(35+106)/252,106/(106+76)


# Linear Discriminant Analysis (LDA)
lda = LDA(store_covariance=True)
X_train, X_test = [M.drop(columns=['intercept'])
                   for M in [X_train, X_test]]
lda.fit(X_train, L_train)
lda.means_
lda.classes_
lda.priors_
lda.scalings_
lda_pred = lda.predict(X_test)
confusion_table(lda_pred, L_test)

lda_prob = lda.predict_proba(X_test)

np.all(
       np.where(lda_prob[:,1] >= 0.5, 'Up','Down') == lda_pred
       )

np.all(
       [lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred
       )
np.sum(lda_prob[:,0] > 0.9)

confusion_table(lda_pred, L_test)


# Quadratic Discriminant Analysis
qda = QDA(store_covariance=True)
qda.fit(X_train, L_train)
qda.means_, qda.priors_
qda.covariance_[0]
qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)
np.mean(qda_pred == L_test)


# Naive Bayes
NB = GaussianNB()
NB.fit(X_train, L_train)
NB.classes_
NB.class_prior_
NB.theta_
NB.var_
X_train[L_train == 'Down'].mean()
X_train[L_train == 'Down'].var(ddof=0)
nb_labels = NB.predict(X_test)
confusion_table(nb_labels, L_test)
NB.predict_proba(X_test)[:5]




# K-Nearest Neighbors
knn1 = KNeighborsClassifier(n_neighbors=1)
X_train, X_test = [np.asarray(X) for X in [X_train, X_test]]
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
confusion_table(knn1_pred, L_test)

(83+43)/252, np.mean(knn1_pred == L_test)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3_pred = knn3.fit(X_train, L_train).predict(X_test)
np.mean(knn3_pred == L_test)

Caravan = load_data('Caravan')
Purchase = Caravan.Purchase
Purchase.value_counts()

feature_df = Caravan.drop(columns=['Purchase'])

scaler = StandardScaler(with_mean=True,
                        with_std=True,
                        copy=True)

scaler.fit(feature_df)
X_std = scaler.transform(feature_df)

feature_std = pd.DataFrame(
                 X_std,
                 columns=feature_df.columns)
feature_std.std()



(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(np.asarray(feature_std),
                            Purchase,
                            test_size=1000,
                            random_state=0)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
np.mean(y_test != knn1_pred), np.mean(y_test != "No")

confusion_table(knn1_pred, y_test)