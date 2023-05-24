# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:52:36 2022

@author: user
"""

import numpy as np
import pandas as pd


df =pd.read_csv(r"C:\Users\user\Desktop\DS\DATA_SET\Regression Data Set\CAR DETAILS FROM CAR DEKHO.csv")
print(df)
df.info()

df = df.reindex(columns = ["name","year","km_driven","fuel","seller_type","transmission","owner","selling_price"])

x = df.iloc[:,:-1]
print(x)
x.info()
x.corr()
y =df.iloc[:,-1]
print(y)

a = pd.get_dummies(x.fuel)
print(a)

b =pd.get_dummies(x.seller_type)
print(b)

c =pd.get_dummies(x.transmission)
print(c)

d =pd.get_dummies(x.owner)
print(d)


new_df = pd.concat([x,a,b,c,d],axis="columns")
print(new_df)

new_df =new_df.drop(["fuel","seller_type","transmission","owner","name"],axis ="columns")
print(new_df)
new_df.corr()
#training and testing data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(new_df,y,test_size=0.3,random_state=500)


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
lbl.score(x_test,y_test)

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



rid2 = Ridge(alpha =50)

rid2.fit(x_train,y_train)

rid2.predict(x_test)
rid2.score(x_test,y_test)
