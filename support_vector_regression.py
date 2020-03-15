#Support Vector Regression
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

#Fitting SVR to Dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

#Predicting results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))) 

#Visualizing Linear Regression Result
plt.scatter(X, y , color ='red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Results for SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()