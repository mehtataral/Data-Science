# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:38:01 2022
train and test Multilinear Regression

@author: user
"""

import numpy as np
import pandas as pd


df =pd.read_csv(r"C:\Users\user\Documents\Taral\DATA_SET\DATA_SET\Regression Data Set\Basic Regression Data\multi_house_dummy.csv")
print(df)

x = df.iloc[:,:-1]
print(x)

y =df.iloc[:,-1]
print(y)

a = pd.get_dummies(x.town)
print(a)

new_df = pd.concat([x,a],axis="columns")
print(new_df)

new_df =new_df.drop("town",axis ="columns")
print(new_df)

#training and testing data

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(new_df,y,test_size=0.3,random_state=5)


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


print(x_train)
print(x_test)
print(y_train)
print(y_test)


from sklearn.linear_model import LinearRegression
lbl =LinearRegression()
lbl.fit(x_train,y_train)


lbl.predict(x_test)
y_test
lbl.score(x_test,y_test)

lbl.score(x_train,y_train)





from sklearn.linear_model import Lasso,Ridge
las1 = Lasso()

las1.fit(x_train,y_train)

las1.predict(x_test)
las1.score(x_test,y_test)



las1 = Lasso(alpha =100)

las1.fit(x_train,y_train)

las1.predict(x_test)
las1.score(x_test,y_test)
 

rid1 = Ridge()

rid1.fit(x_train,y_train)

rid1.predict(x_test)
rid1.score(x_test,y_test)



rid2 = Ridge(alpha =10)

rid2.fit(x_train,y_train)

rid2.predict(x_test)
rid2.score(x_test,y_test)



