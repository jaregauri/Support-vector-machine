# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:11:19 2020

@author: Gauri
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("ps.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#reshape for standardscaler class
y = y.reshape(len(y), 1)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_Y = StandardScaler()
Y = sc_Y.fit_transform(y)

#TRAINING

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#predict
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#visualise
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVM)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#smooth curve
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color = 'red')
plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVM)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



