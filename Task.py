# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('WorkingTable1.csv')
X = dataset.iloc[:, 8:33].values
y = dataset.iloc[:, 33].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Backward Elimination on linear regression
import statsmodels.formula.api as sm
"To count in the constant on linear regression"
X = np.append(arr = np.ones((14468,1)).astype(int), values = X, axis = 1)
"Remove the one with the largest P value"
X_opt = X [:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 13, 14, 19, 20, 21, 22, 23, 26]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 13, 14, 19, 20, 21, 22, 23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 10, 11, 14, 19, 20, 21, 22, 23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 11, 14, 19, 20, 21, 22, 23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 7, 11, 19, 20, 21, 22, 23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 5, 6, 11, 19, 20, 21, 22, 23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X [:, [0, 1, 2, 4, 6, 11, 19, 20, 21, 22, 23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()






