import numpy as np
import pandas as pd

df =pd.read_csv(r"D:\Taral\DATA_SET\DATA_SET\classification data\Churn_Modelling.csv")
print(df)
print(df.info())
print(df.Geography)
print(df.NumOfProducts)

a = df.corr()#heat map    
print(a)
x = df.iloc[:,:-1]
print(x)

y = df.iloc[:,-1]
print(y)

x.isnull().any()
x.isnull().sum()
y.isnull().any()
y.isnull().sum()

x.fillna(df.Age.mean(),inplace =True)
x.isnull().any()
x.isnull().sum()

x.drop(["Surname","CustomerId","RowNumber"],axis = "columns",inplace =True)
print(x)

x.isnull().any()
x.isnull().sum()
y.isnull().sum()

from sklearn.preprocessing import LabelEncoder# convert string to number
lbl = LabelEncoder()
x["Gender"] = lbl.fit_transform(x.Gender)
x["Geography"] = lbl.fit_transform(x.Geography)
print(x)

x.EstimatedSalary

df.nlargest(5,"EstimatedSalary")

df.nsmallest(5,"EstimatedSalary")


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=4)

print(len(x_train))
print(x_test)
print(y_train)
print(y_test)   

from sklearn.preprocessing import MinMaxScaler#large data converted into (0 to 1)
ms = MinMaxScaler()
df["EstimatedSalary"] = ms.fit_transform(df[["EstimatedSalary"]])
print(df)

x_train = ms.fit_transform(x_train)
print(x_train)

x_df1= pd.DataFrame(x_train)
print(x_df1)

x_test = ms.fit_transform(x_test)
print(x_test)

#support vector machine
#it is used for smaller data set

# As the value of ‘c’ increases the model gets overfits. vice versa
# γ : Gamma (used only for RBF kernel) it is used only when data  set is non linear

# Pros:
# It is really effective in the higher dimension.
# Effective when the number of features are more than training examples.
# Best algorithm when classes are separable
# The hyperplane is affected by only the support vectors thus outliers have less impact.
# SVM is suited for extreme case binary classification.
# cons:
# For larger dataset, it requires a large amount of time to process.
# Does not perform well in case of overlapped classes.
# Selecting, appropriately hyperparameters of the SVM that will allow for sufficient 
# generalization performance.
# Selecting the appropriate kernel function can be tricky.

from sklearn.svm import SVC
sr =SVC()#by default rbf

sr =SVC(kernel = "linear")#it takes too much time
sr =SVC(kernel = "poly",degree = 3,C=100,)
sr =SVC(kernel = "rbf",C =50,gamma=10)#Radial basis function kernel (RBF)/ Gaussian Kernel
sr =SVC(kernel = "sigmoid",C =40)
# sr =SVC(kernel = "precomputed")#Precomputed matrix must be a square matrix

sr.fit(x_train,y_train)
sr.score(x_test,y_test)

sr.score(x_train,y_train)

#Decision Tree
# Like entropy, Gini impurity is also a measure of calculating the 
# node impurity with the same mathematical range [0–1].
# Where 1 means maximum impurity and 0 means the least.

# Entropy by definition is a lack of order or predictability. It is the measure of impurity in a bunch of examples. 
# The node is the purest if it has the instances of only one class.

# You can also use Gini impurity instead of Entropy. 
# Even though, the results wouldn’t differ much whether you use one or the other. 
# But computationally Gini impurity is slightly faster than entropy. 
# Entropy produces more balanced trees whereas Gini impurity tends to separate the majority class into its own branch.
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt
from sklearn import tree
dt = DecisionTreeClassifier(criterion ="gini")
dt = DecisionTreeClassifier(criterion ="gini",min_samples_split=15)


dt.fit(x_train,y_train)
dt.score(x_test,y_test)
dt.score(x_train,y_train)
# feature = ["CreditScore","Geography","Gender","Age","Tenure","Balance"]
# tree.plot_tree(dt,feature_names = feature)
AA= dt.predict(x_test)
AA
y_test

dt = DecisionTreeClassifier(criterion ="entropy")
dt.fit(x_train,y_train)
dt.score(x_test,y_test)
dt.score(x_train,y_train)
dt.predict(x_test)
y_test


#Random Forest Tree
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf = RandomForestClassifier(n_estimators=1500,oob_score = True)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
print(rf.oob_score_)

from sklearn.metrics import accuracy_score
y_pred = rf.predict(x_test)
print(y_pred)
accuracy_score(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier #out of bag
rf = RandomForestClassifier(n_estimators=1000,criterion="entropy",oob_score = True)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
rf.score(x_train,y_train)

rf.oob_score()

#KNN
from sklearn.neighbors import KNeighborsClassifier
kc = KNeighborsClassifier(n_neighbors=5)
kc.fit(x_train,y_train)
kc.score(x_test,y_test)
kc.score(x_train,y_train)

#NaiveBayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
mnb.score(x_test,y_test)
mnb.score(x_train,y_train)

from sklearn.naive_bayes import GaussianNB
mnb = GaussianNB()
mnb.fit(x_train,y_train)
mnb.score(x_test,y_test)
mnb.score(x_train,y_train)

from sklearn.naive_bayes import BernoulliNB
mnb = BernoulliNB()
mnb.fit(x_train,y_train)
mnb.score(x_test,y_test)
mnb.score(x_train,y_train)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
modelLR=LogisticRegression()
modelLR.fit(x_train,y_train)
modelLR.score(x_test,y_test)
modelLR.score(x_train,y_train)