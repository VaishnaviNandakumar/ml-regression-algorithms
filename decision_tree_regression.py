#Decision Tree Regression 1D

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Feature Scaling (Required here cause SVR is a less common module)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting Decision Tree Reegressor to Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Predicting results
y_pred =regressor.predict(([[6.5]]))
#Visualizing Decision Tree Regression Result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y , color ='red')
plt.plot(X_grid, regressor.predict((X_grid)), color = 'blue')
plt.title('Results for  Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()