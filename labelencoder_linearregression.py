# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:06:40 2022
Label Encoder Multilinear Regression

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv(r"C:\Users\user\Documents\Taral\DATA_SET\DATA_SET\Regression Data Set\Basic Regression Data\multi_house_dummy.csv")
print(df)
df = df.reindex(columns=["area","town","price"])
print(df)

x = df.iloc[:,:-1]
print(x)

y =df.iloc[:,-1:]
print(y)

plt.scatter(x.area, y)
plt.show()

#LabelEncoder converts strings inot numbers
from sklearn.preprocessing import LabelEncoder
lbl =LabelEncoder()
x.town = lbl.fit_transform(x.town)
print(x)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
reg.predict(x)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pr = PolynomialFeatures(degree =3)
x_poly = pr.fit_transform(x)
reg = LinearRegression()
reg.fit(x_poly,y)

#lasso regression 
#ridge regression
#polynomail regression --> scatter

print(reg.coef_)
print(reg.intercept_)

reg.predict(x_poly)
reg.score(x_poly,y)

reg.predict([[3600]])
reg.predict([[2600]])
reg.predict([[4600]])

reg.predict([[1,2600]])
reg.predict([[2,2600]])
reg.score(x,y)

plt.scatter(x.area, y)
plt.plot(x.area,reg.predict(x),'rD:')
plt.show()


from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV
model1=Ridge(alpha=5)
model1.fit(x,y)
model1.score(x,y)
print(model1.coef_)#coefiicent
print(model1.intercept_)
model1.predict([[2600]])
model1.predict([[3600]])
model1.predict([[4600]])
model1.predict([[2000]])

model2=Lasso(alpha=6)
model2.fit(x,y)
model2.score(x,y)
print(model2.coef_)
print(model2.intercept_)


m1=Ridge()
parameters={'alpha':[1e-15,1e-18,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

lasso_regressor=GridSearchCV(m1,parameters,cv=5)
lasso_regressor.fit(x,y)
lasso_regressor.score(x,y)

print(model2.coef_)

# from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y,ypred)
# rmse= np.sqrt(mse)
# 