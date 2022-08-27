# Polynomial Regression

# Data preprocessing
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Building Linear Regressiion Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Building the Polynomial Regressor 
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree=2)
x_polynomial = polynomial.fit_transform(X)
polynomial_regressor = LinearRegression()
polynomial_regressor.fit(x_polynomial, y)