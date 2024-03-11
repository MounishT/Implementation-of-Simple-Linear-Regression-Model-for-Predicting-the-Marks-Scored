# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values and import linear regression from sklearn.
3. Assign the points for representing in the graph.
4. Predict the regression for marks by using the representation of the graph and compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by:T MOUNISH 
RegisterNumber:212223240098  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("CSVs/student_scores.csv")
df.head()
df.tail()
X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)
plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("MSE : ",mean_squared_error(Ytest,Ypred))
print("MAE : ",mean_absolute_error(Ytest,Ypred))
print("RMSE : ",np.sqrt(mse))

```

## Output:
df.head()

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/740dca1e-8c66-4be5-800b-852af7397a1e)

df.tail()

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/09827cc8-a000-4b44-8570-54549fd6db5e)

Array values of X

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/4998f9f8-d802-4816-a83c-95a5e1e4ca4e)

Array values of Y

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/afa6b95d-b43d-471a-89b8-b89d3e914264)

Values of Y prediction

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/56a38553-87a1-4e57-b7a6-e1d81a1c629d)

Array values of Y test

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/dc6c31e2-70a6-4a53-a914-d6766dcd5123)

Training Set Data

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/61ab9ab9-5268-46c1-8039-3f33141c6907)

Test Set Data

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/afad7a76-be2a-4aa5-bf69-4322cf19602c)

Values of MSE, MAE and RMSE

![image](https://github.com/saiganesh2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742342/30a68ea1-f212-4f19-8422-3c4a3966a74b)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
