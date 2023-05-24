# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:47:54 2023

@author: user
"""
import pandas as pd
#DUPLICATE 
#NAN VALUE 
#FILL VALUE
#INSIGHT 
import numpy as np
df = pd.read_excel(r"C:\Users\user\Downloads\archive\superstore.xls")
df

#***************************Outliers using percentile techniques************************************
df.describe()
df["Profit"].describe()


df.Profit.head(5)
df.Profit.nlargest()
df.Profit.nsmallest()
df.Profit.median()

df.Profit.quantile(0.3)
df.Profit.quantile(0.5)
df.Profit.quantile(0.7)
df.Profit.quantile(0.9)
df.Profit.quantile(0.95)
df.Profit.quantile(0.995)
df.Profit.quantile(0.005)

x = df["Profit"].quantile(0.995) #10 0%
x
y= df["Profit"].quantile(0.005)#0%
y
df

df1 = df[(df.Profit <= x) & (df.Profit >=y)]
df1

df1.describe()
df1.median()


# df1 = df[(df.Profit <= x)]
df1.Profit.min()
df1.Profit.max()

print(df1.to_string())

#****

df.Balance
i = df.Profit.quantile(1)
i
j = df.Profit.quantile(0)
j

df1 = df[(df.Profit < i) & (df.Profit >j)]
df1

#***************************Outliers using IQR InterQuantile Range
x1 = df.describe()

# on EstimatedSalary
# IQR = Q3-Q1 = 98385.89

# standard deviation
# mean
# median 
# mode

Q1 =df.Profit.quantile(0.25)
Q3 =df.Profit.quantile(0.75)
IQR = Q3-Q1
IQR
lower_limit = Q1 -1.5*IQR#3
lower_limit
upper_limit = Q3+1.5*IQR
upper_limit
out = df[(df.Profit<lower_limit)|(df.Profit>upper_limit)]

import seaborn as sns

sns.boxplot(df.Profit)
sns.boxplot(df.Sales)
sns.boxplot(df.Quantity)
 