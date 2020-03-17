#Random Forest Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#Fitting Decision Tree Reegressor to Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,y)

#Predicting results
y_pred =regressor.predict(([[6.5]]))
#Visualizing Random Foresr Regression Result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y , color ='red')
plt.plot(X_grid, regressor.predict((X_grid)), color = 'blue')
plt.title('Results for  Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
