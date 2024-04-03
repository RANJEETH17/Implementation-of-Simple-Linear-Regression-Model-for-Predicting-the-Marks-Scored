# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ranjeeth B K
RegisterNumber:21222040132
*/

```python
import pandas as pd
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1.csv')
df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.xlabel('Y')
X=df.iloc[:,0:1]
Y=df.iloc[:,-1]
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.xlabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
```

## Output:

# 1)HEAD:
![image](https://github.com/RANJEETH17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120718823/1af3f0ff-b9cb-46b6-a748-b48bb02b6e93)
# 2)GRAPH OF PLOTTED DATA:
![image](https://github.com/RANJEETH17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120718823/5aff7234-94c1-4cc4-bb96-e1dbe25e3498)
# 3)TRAINED DATA:
![image](https://github.com/RANJEETH17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120718823/8b82f303-1b9b-4936-bc19-5e2393b84e4d)
# 4)LINE OF REGRESSION:
![image](https://github.com/RANJEETH17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120718823/7e66ac5c-e6ea-48be-a917-76bdf617ad9e)
# 5)COEFFICIENT AND INTERCEPT VALUES:
![image](https://github.com/RANJEETH17/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120718823/3027e5ed-fde3-443a-ac61-e263adda29b3)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
