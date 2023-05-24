# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:10:59 2022

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x ={
    "area":[2600,3000,3200,3600,4000],
    "price":["55000000","56500000","61000000","59500000","76000000"]
    }

print(x)
df =pd.DataFrame(x)
print(df)
# y ={
#     "area":[2600,3000,3200,3600,4000,2800,3400,3800,3900,4900]
#     }

# print(y)
# df =pd.DataFrame(y)
# print(df)
# df.to_csv(r"C:\Users\user\Desktop\DS\DATA_SET\house_pred.csv",index =False)
# df.to_csv(r"C:\Users\user\Desktop\DS\DATA_SET\house_area.csv",index =False)


# df =pd.read_csv(r"C:\Users\user\Desktop\DS\DATA_SET\house_pred.csv")
# print(df)

x = df.iloc[:,0]
y = df.iloc[:,-1]
plt.xlabel("Price")
plt.ylabel("Area")
plt.scatter(df.price,df.area,marker="+")
plt.plot(df.price,df.area)
plt.show()

from sklearn.linear_model import LinearRegression
reg =LinearRegression()
# reg.fit([[x],[y]])
reg.fit(df[['area']],df.price)
# reg.fit(x[x['area']],y.price) #not valid ..used xtrain and ytrain
reg.coef_
reg.intercept_
x =reg.predict([[3300]])
reg.predict([[2600]])
reg.predict([[4500]])


x =reg.predict(df)#only area is mentioned in df 
df['price'] = x #store that price values in price
print(df)    

plt.xlabel("Price")
plt.ylabel("Area")
plt.scatter(df.price,df.area,marker="+")
plt.plot(df.area,reg.predict(df[['area']]),color ="green",marker = "o")
plt.show()

# from sklearn.metrics import accuracy_score
# acc =accuracy_score((2600),(4500))#not valid ..used xtrain and ytrain
# print(acc)
