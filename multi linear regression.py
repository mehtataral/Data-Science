# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:04:12 2022

@author: user
"""
import pandas as pd
import numpy as np
# x ={
#     "area":[2600,3000,3200,3600,4000],
#     "bedroom":[3,4,np.nan,3,5],
#     "age_house":[20,15,np.nan,30,8],
#     "price":["55000000","56500000","61000000","59500000","76000000"]
#     }
# print(x)

# df =pd.DataFrame(x)
# print(df)
# df.to_csv(r"C:\Users\user\Desktop\DS\DATA_SET\multi_house_para.csv",index =False)

df =pd.read_csv(r"C:\Users\user\Documents\Taral\DataScience\Regression\multi_house_para.csv")
print(df)


df["bedroom"] =df['bedroom'].fillna(df.bedroom.mean())
print(df)
df["age_house"] = df.age_house.fillna(df.age_house.mean())
print(df)


df.bedroom = df.bedroom.astype(int)
df.age_house = df.age_house.astype(int)
print(df)

x =df.iloc[:,:-1]
y =df.iloc[:,-1]
print(x)
print(y)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
reg.predict([[2600,3,20]])#area, bedroom , age_house
reg.predict([[2600,3,10]])
reg.predict([[2600,3,1]]) 
reg.predict([[600,1,30]])

print(reg.score(x,y))

reg.fit(df[["area","bedroom","age_house"]],df.price)
reg.predict([[2600,3,20]])#area, bedroom , age_house
reg.predict([[2600,3,10]])
reg.predict([[2600,3,1]])

reg.predict([[3600,3,30]])









