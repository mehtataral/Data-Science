import numpy as np
import pandas as pd


df =pd.read_csv(r"C:\Users\user\Documents\Taral\DATA_SET\DATA_SET\Regression Data Set\Mumbai1.csv")
print(df)
df.info()
df.drop("Unnamed: 0",axis ="columns",inplace =True)

x = df.iloc[:,1:]#input
print(x)
x.info()
x.corr()
y =df.iloc[:,0]#output
print(y)

corr = x.corr()
#training and testing data

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x.Location = lb.fit_transform(x.Location)
print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=500)


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


las1 = Lasso(alpha =1000,selection="random")

las1.fit(x_train,y_train)

las1.predict(x_test)
las1.score(x_test,y_test)
   

rid1 = Ridge()

rid1.fit(x_train,y_train)

predict1=rid1.predict(x_test)
rid1.score(x_test,y_test)

# dff = pd.DataFrame(predict1)
# dff
# dff["actual_value"] =y_test
# dff

rid2 = Ridge(alpha =50)

rid2.fit(x_train,y_train)

rid2.predict(x_test)
rid2.score(x_test,y_test)
rid2.score(x_train,y_train)

