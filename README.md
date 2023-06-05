# Implementation-of-SVM-For-Spam-Mail-Detection

## Aim:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Smriti M
RegisterNumber: 212221040157
*/

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## Result output

![s1](https://github.com/SmritiManikand/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113674204/e19527e9-4380-4b8d-8af3-96685f661789)

## data.head()

![s2](https://github.com/SmritiManikand/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113674204/25e34fea-df02-491e-a5ae-23f3beea29d4)

## data.info()

![s3](https://github.com/SmritiManikand/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113674204/6e89401b-0fbf-4184-a80b-348a393866e9)

## data.isnull().sum()

![s4](https://github.com/SmritiManikand/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113674204/23c36115-56f6-4b61-ab46-273bc6d24e99)

## Y_prediction value

![s5](https://github.com/SmritiManikand/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113674204/3eeb308d-90ed-489e-b0d4-7d3369f8aedc)

## Accuracy value

![s6](https://github.com/SmritiManikand/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113674204/ac1c854b-8df7-4db3-a8be-7bd299fec0e9)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
