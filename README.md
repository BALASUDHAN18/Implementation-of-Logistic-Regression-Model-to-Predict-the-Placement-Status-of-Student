# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BALASUDHAN P
RegisterNumber: 212222240017

import pandas as pd
data = pd.read_csv('dataset/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:, :-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/97e41d4d-c804-455d-a82d-020afa8d2b0e)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/a4f09604-f420-499a-b9c0-f834f9e29641)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/9af66100-cb68-4e0c-aa6d-d54a40b16bf3)
![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/f5b91202-f0df-47f4-89b7-f854d022f22c)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/9402a5f9-dba7-43f2-8522-52ef24e9f96c)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/9ac4975a-00e1-4d11-9481-60ece2adc905)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/b5c7f27d-472e-431e-9b4e-13af3269efd8)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/a4a46f14-b795-4e7d-915c-1697d831a65e)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/154405bc-a0a8-43f9-9c62-63b6524bd977)

![image](https://github.com/BALASUDHAN18/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118807740/46411c42-25dc-4c4f-bedc-46dab3f38e6c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
