# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:26:46 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:29:51 2020
LASSO
@author: Trainer 1
"""
x =(1,2,3)
lst =list(x)
print(lst)
lst.append(55)
lst
x =tuple(lst)
import pandas as pd
import numpy as np
df=pd.read_csv('Advertising.csv')
df.head()
df.drop('SR',axis=1,inplace=True)

from sklearn.linear_model import LinearRegression
x=df.drop('SALE',axis=1)
y=pd.DataFrame(df.SALE)
model1=LinearRegression()
model1.fit(x,y)
print(model1.coef_)
ki7i

from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV
model2=Ridge(alpha=700)
model2.fit(x,y)
print(model2.coef_)


model3=Lasso(alpha=6)
model3.fit(x,y)
model3.coef_
m1=Ridge()
parameters={'alpha':[1e-15,1e-18,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

lasso_regressor=GridSearchCV(m1,parameters,cv=5)
lasso_regressor.fit(x,y)
print(model3.coef_)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


89,102,88,0  :Lasso n rigid 
y=89*555676+0*23454+

higer no feature n data : processing time n compution cost

when we say Processing Time  incerse : n no calct happen backen  
foreg :
    y=m1x1+m2x2+m3x3+m4x4 +c
    m1=9,m2=6,m3=4,m4=2, x1=100,x2=350,x3=55,x4=10
what if i have  m1=220,m2=300,m4=267,m3=200 then processing time then we need to find a method to decrese the values  how ???
 Lasso n Rigid : decrese the values 
 

Featureselection : chisquare test 
