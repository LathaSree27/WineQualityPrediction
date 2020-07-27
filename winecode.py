"""
Prediction of wine quality using machine learning in python

"""

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#reading dataset
raw_data=pd.read_csv('file:///C:/Users/Lathasree Reddy/Documents/winequality-white.csv',sep=';')
print("Shape of raw data :",raw_data.shape,"\n")

#identifying duplicates
dup_data=raw_data.duplicated()
print("Number of duplicate rows =",sum(dup_data),"\n")

#removing duplicates
data=raw_data.drop_duplicates()
print("Shape of data after removing duplicates :",data.shape,"\n")

data.rename(columns={'fixed acidity':'fixed_acidity','volatile acidity':'volatile_acidity',\
                     'citric acid':'citric_acid','residual sugar':'residual_sugar',\
                     'free sulfur dioxide':'free_sulfur_dioxide',\
                     'total sulfur dioxide':'total_sulfur_dioxide'},inplace=True)

#printing first 5 data instances
Head=data.head()
print(data.head())
print()

#checking for missing values
print(data.isnull().sum())
print()
#there are no missing values

#data set description
data_info=data.describe()
print(data_info)

#checking the correlation between attributes
data_corr=data.corr()
print(data_corr)

#spliting the data set
train,test=tts(data,test_size=0.2)

y=train['quality']
cols=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",\
      "free_sulfur_dioxide","total_sulfur_dioxide","pH","sulphates","alcohol"]
X=train[cols]

#model formation
reg=linear_model.LinearRegression()
model=reg.fit(X,y)

coef=reg.coef_
print("Coefficients of the linear equation : \n",coef,"\n")
intercept=reg.intercept_
print("Y-intercept :",intercept)
print()

y_train_pred=reg.predict(X)
print("In sample Root mean square error: %.2f"%mean_squared_error(y,y_train_pred)**0.5)
print()

y_test=test['quality']
X_test=test[cols]

y_test_pred=reg.predict(X_test)
print("Out sample Root mean square error: %.2f"%mean_squared_error(y_test,y_test_pred)**0.5)
print()

#unknown sample
import numpy as np
a=np.array([12,0.5,1.4,56,0.2,205,380,3.1,0.3,7.1]).reshape(1,-1)
quality1=reg.predict(a)
print(quality1)


#unknown sample from user
from array import array
arr=array('f',[])
x=float(input("Enter the value of fixed acidity(range(3 to 15))"))
arr.append(x)
x=float(input("Enter the value of volatile acidity(range(0 to 1))"))
arr.append(x)
x=float(input("Enter the value of citric acidity(range(0 to 2))"))
arr.append(x)
x=float(input("Enter the value of residual sugar(range(0 to 100))"))
arr.append(x)
x=float(input("Enter the value of chlorides(range(0 to 0.5))"))
arr.append(x)
x=float(input("Enter the value of free sulphur dioxide(range(0 to 300))"))
arr.append(x)
x=float(input("Enter the value of total sulphur dioxide(range(0 to 500))"))
arr.append(x)
x=float(input("Enter the value of pH(range(2 to 4))"))
arr.append(x)
x=float(input("Enter the value of sulphates(range(0 to 1))"))
arr.append(x)
x=float(input("Enter the value of alcohol(range(5 to 15))"))
arr.append(x)


ar=np.asarray(arr)
ar=ar.reshape(1,-1)

quality2=reg.predict(ar)
print()
print("Quality of your wine sample is",quality2)
 

