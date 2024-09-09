# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot
2. Trace the best fit line and calculate the cost function
3. Calculate the gradient descent and plot the graph for it
4. Predict the profit for two population sizes.
  
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: T. Gayathri
RegisterNumber:  212223100007

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regresssion(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        prediction = (X).dot(theta).reshape(-1,1)
        errors=(prediction-y).reshape(-1,1)
        theta -= learning_rate * (1/ len(X1)) * X.T.dot(errors)
    return theta
data =pd.read_csv("50_Startups.csv",header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler= StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regresssion(X1_Scaled, Y1_Scaled)

new_data = np.array([165343.2,136897.8,471784.1]).reshape(-1,1)
new_Scaler = scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaler),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

*/
```

## Output:

 ## Head
 ![image](https://github.com/user-attachments/assets/12186dc0-c93b-439d-ae9a-7aef825cd795)

## Linear Regression
![Screenshot 2024-09-09 100756](https://github.com/user-attachments/assets/5d50b969-5cdd-4933-bf0c-f3fa25c10260)
![Screenshot 2024-09-09 100813](https://github.com/user-attachments/assets/a2065eae-67c0-432f-9cd2-de35411804e3)
![Screenshot 2024-09-09 100826](https://github.com/user-attachments/assets/77a17df9-749b-4680-ae3e-7f8271ddb355)
![Screenshot 2024-09-09 100845](https://github.com/user-attachments/assets/4d3138d5-2958-4b56-926d-ccc73d2dc1a8)
![Screenshot 2024-09-09 100845](https://github.com/user-attachments/assets/02fe74af-1cb6-4e58-abb5-dbb7e53ba4a7)
![Screenshot 2024-09-09 100921](https://github.com/user-attachments/assets/bc4d84cf-28c1-43db-b95c-d842ad906e35)
![Screenshot 2024-09-09 100930](https://github.com/user-attachments/assets/b07324eb-d795-4d91-8659-357048949678)
![Screenshot 2024-09-09 100942](https://github.com/user-attachments/assets/b0ce6b33-1460-4623-be12-3cebae28132e)
![Screenshot 2024-09-09 100949](https://github.com/user-attachments/assets/f5a6bfc5-d5e2-4c5d-84e0-55c92d447830)

## Prediction

![Screenshot 2024-09-09 111317](https://github.com/user-attachments/assets/7d62395d-5159-406b-93fe-5e87cdd1e507)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
