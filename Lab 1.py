"""
Chapter 2 - Question 4:
You will now think of some real-life applications for statistical learn-
ing.
(a) Describe three real-life applications in which classification might
be useful. Describe the response, as well as the predictors. Is the
goal of each application inference or prediction? Explain your
answer.
(b) Describe three real-life applications in which regression might
be useful. Describe the response, as well as the predictors. Is the
goal of each application inference or prediction? Explain your
answer.
(c) Describe three real-life applications in which cluster analysis
might be useful.

Answer:
==============
(a) Applications of Classification:
1. Email spam detection:
   - Response: The classification of emails as "spam" or "not spam".
   - Predictors: Features such as the presence of certain keywords, the sender's email address, the frequency of links, and the email's metadata.
   - Goal: Prediction. The aim is to predict whether a new incoming email is spam based on learned patterns from previous emails.
   
2. Medical Diagnosis:
   - Response: Classifying patients as having a specific disease or not (e.g., diabetes, cancer).
   - Predictors: Patient data such as age, symptoms, medical history, lab test results, and genetic information.
   - Goal: Inference. The goal is to understand the relationship between predictors and the disease, which can help in understanding risk factors and improving treatment strategies.

3. Credit Scoring:
   - Response: Classifying individuals as "creditworthy" or "not creditworthy".
   - Predictors: Financial history, income level, existing debts, credit utilization ratio, and demographic information.
   - Goal: Prediction. The objective is to predict the likelihood of an individual defaulting on a loan based on their financial profile.

4. Meteological Event Classification (whether it will rain or not):
    - Response: Classifying weather conditions as "rain" or "no rain".
    - Predictors: Atmospheric data such as humidity, temperature, wind speed, and historical weather patterns.
    - Goal: Prediction. The aim is to predict future weather conditions based on current and historical data.

    
(b) Applications of Regression: 
1. House Price Prediction:
   - Response: The price of a house.
   - Predictors: Features such as the size of the house, number of bedrooms and bathrooms, location, age of the property, and local amenities.
   - Goal: Prediction. The objective is to predict the selling price of a house based on its characteristics.
2. Sales Forecasting:
   - Response: Future sales figures for a product.
   - Predictors: Historical sales data, marketing spend, seasonality, economic indicators, and competitor actions.
   - Goal: Prediction. The aim is to forecast future sales based on past trends and influencing factors.
3. Health Care Costs Estimation:
   - Response: The total healthcare costs for a patient.
   - Predictors: Patient demographics, medical history, treatment plans, and lifestyle factors.
   - Goal: Prediction. The goal is to estimate future healthcare costs based on various predictors to help in budgeting and resource allocation.
4. Energy Consumption Forecasting:
   - Response: The amount of energy consumed (e.g., electricity, gas).
   - Predictors: Factors such as historical energy usage, weather conditions, time of year, and household characteristics.
   - Goal: Prediction. The aim is to predict future energy consumption based on past usage patterns and external factors.


(c) Applications of Cluster Analysis:
1. Customer Segmentation: Grouping customers based on purchasing behavior, preferences, and demographics.
   - Use: Businesses can tailor marketing strategies and product offerings to different customer segments.
   - Goal: Understanding customer diversity and targeting specific groups effectively.
2. Image Segmentation: Grouping pixels in an image based on color, texture, or intensity.
   - Use: Used in computer vision applications such as object detection, facial recognition, and medical imaging.
   - Goal: To identify and isolate different objects or regions within an image for further analysis.
3. Social Network Analysis: Grouping individuals in a social network based on their interactions, relationships, or shared interests.
   - Use: Helps in understanding community structures, influence patterns, and information flow within networks.
   - Goal: To identify clusters of individuals who are closely connected, which can inform strategies for engagement or information dissemination.

"""


# Question 7 from chap 2:

import numpy as np
import pandas as pd

# Dataset
data = pd.DataFrame({
    'Obs': [1, 2, 3, 4, 5, 6],
    'X1': [0, 2, 0, 0, -1, 1],
    'X2': [3, 0, 1, 1, 0, 1],
    'X3': [0, 0, 3, 2, 1, 1],
    'Y': ['Red', 'Red', 'Red', 'Green', 'Green', 'Red']
})

# Test point
test_point = np.array([0, 0, 0])

# Compute Euclidean distances
def euclidean_distance(row):
    return np.linalg.norm(np.array([row['X1'], row['X2'], row['X3']]) - test_point)

data['Distance'] = data.apply(euclidean_distance, axis=1)

# Sort by distance
sorted_data = data.sort_values(by='Distance').reset_index(drop=True)

# Show results
print("Sorted data with distances to test point (0,0,0):\n")
print(sorted_data[['Obs', 'X1', 'X2', 'X3', 'Y', 'Distance']])

# K = 1 prediction
k1_prediction = sorted_data.iloc[0]['Y']
print(f"\nPrediction with K=1: {k1_prediction}")

# K = 3 prediction (majority vote)
k3_neighbors = sorted_data.iloc[:3]
k3_prediction = k3_neighbors['Y'].mode()[0]
print(f"Prediction with K=3: {k3_prediction}")



"""
Question 4 from Chapter 3:
a) We expect the training RSS for the cubic regression to be lower than that of the linear regression, because it has more flexibility and can better fit the training data — even if those extra terms are not needed.
b) We expect the test RSS to be lower for the linear regression, because the cubic regression may overfit to noise in the training data, increasing its test error.
c) We expect the training RSS to be lower for the cubic regression, because it has more parameters to better fit both linear and nonlinear patterns.
d) There is not enough information to tell which test RSS will be lower. It depends on how nonlinear the true relationship is. The cubic model might perform better if the nonlinearity is substantial, but might overfit if the nonlinearity is minimal.
"""



# Question 9 from Chap 3 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load and clean data
auto_df = pd.read_csv("Auto.csv")
auto_df['horsepower'] = pd.to_numeric(auto_df['horsepower'], errors='coerce')
auto_df.dropna(inplace=True)

# (a) Scatterplot Matrix
sns.pairplot(auto_df.drop(columns=['name']))
plt.suptitle("Scatterplot Matrix of Auto Dataset", y=1.02)
plt.show()

# (b) Correlation Matrix
correlation_matrix = auto_df.drop(columns=['name']).corr()
print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# (c) Multiple Linear Regression
auto_df_clean = auto_df.drop(columns=['name'])
formula = 'mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin'
model = ols(formula, data=auto_df_clean).fit()
print(model.summary())

# (c.i) ANOVA
anova_results = anova_lm(model)
print("ANOVA Results:")
print(anova_results)

# (d) Diagnostic Plots
residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(6, 4))
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

sm.graphics.influence_plot(model, criterion="cooks")
plt.show()

# (e) Interaction Terms
interaction_model = ols('mpg ~ weight * horsepower + year + cylinders + displacement + acceleration + origin', data=auto_df_clean).fit()
print(interaction_model.summary())

# (f) Transformations
model_log_disp = ols('mpg ~ np.log(displacement) + horsepower + weight + acceleration + year + origin + cylinders', data=auto_df_clean).fit()
print("Log(displacement) model:")
print(model_log_disp.summary())

model_sqrt_weight = ols('mpg ~ np.sqrt(weight) + horsepower + displacement + acceleration + year + origin + cylinders', data=auto_df_clean).fit()
print("Sqrt(weight) model:")
print(model_sqrt_weight.summary())

model_year_squared = ols('mpg ~ year + I(year**2) + horsepower + weight + displacement + acceleration + origin + cylinders', data=auto_df_clean).fit()
print("Year + Year^2 model:")
print(model_year_squared.summary())





# Question 14 from chapter 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# (a) Generate synthetic data
rng = np.random.default_rng(10)
x1 = rng.uniform(0, 1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# (b) Correlation and scatterplot
correlation = np.corrcoef(x1, x2)[0, 1]
print(f"Correlation between x1 and x2: {correlation:.2f}")

plt.figure(figsize=(6, 4))
sns.scatterplot(x=x1, y=x2)
plt.title(f"Scatterplot of x1 vs x2 (Correlation: {correlation:.2f})")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.tight_layout()
plt.show()

# (c) Multiple regression y ~ x1 + x2
X_full = sm.add_constant(df[['x1', 'x2']])
model_full = sm.OLS(df['y'], X_full).fit()
print("\nMultiple Regression (y ~ x1 + x2):")
print(model_full.summary())

# (d) Regression y ~ x1 only
X_x1 = sm.add_constant(df[['x1']])
model_x1 = sm.OLS(df['y'], X_x1).fit()
print("\nSimple Regression (y ~ x1):")
print(model_x1.summary())

# (e) Regression y ~ x2 only
X_x2 = sm.add_constant(df[['x2']])
model_x2 = sm.OLS(df['y'], X_x2).fit()
print("\nSimple Regression (y ~ x2):")
print(model_x2.summary())

# (f) Interpretation:
print("\nNote: Individually, both x1 and x2 are significant.")
print("In the combined model, collinearity makes x2 appear insignificant.")

# (g) Add mismeasured outlier observation
x1_new = np.concatenate([x1, [0.1]])
x2_new = np.concatenate([x2, [0.8]])
y_new = np.concatenate([y, [6]])
df_new = pd.DataFrame({'x1': x1_new, 'x2': x2_new, 'y': y_new})

# Re-fit all models
X_full_new = sm.add_constant(df_new[['x1', 'x2']])
model_full_new = sm.OLS(df_new['y'], X_full_new).fit()

X_x1_new = sm.add_constant(df_new[['x1']])
model_x1_new = sm.OLS(df_new['y'], X_x1_new).fit()

X_x2_new = sm.add_constant(df_new[['x2']])
model_x2_new = sm.OLS(df_new['y'], X_x2_new).fit()

print("\nWith Outlier Added:")
print("\nFull Model with Outlier:")
print(model_full_new.summary())

print("\nModel (y ~ x1) with Outlier:")
print(model_x1_new.summary())

print("\nModel (y ~ x2) with Outlier:")
print(model_x2_new.summary())

# Influence plot for full model with outlier
sm.graphics.influence_plot(model_full_new, criterion="cooks", size=1)
plt.title("Influence Plot with Outlier (Full Model)")
plt.tight_layout()
plt.show()


"""
In the full model, the presence of the outlier increased multicollinearity sensitivity: x1 is now marginally insignificant, while x2 became more significant.

The influence plot shows this new point (#100) has very high leverage and a large residual → it is both an outlier and a high-leverage point.
"""

