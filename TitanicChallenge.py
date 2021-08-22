import pandas as pd
import numpy as np
from sklearn import preprocessing as prepo
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import style
import matplotlib.pyplot as plt

#Read the Data Set
train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")
test_Passenger_Ids =  test_dataset["PassengerId"]
print("Gathering the Dataset")

#Remove columns that are not required
train_dataset.drop(["Name","Ticket","Fare","Cabin"],inplace = True,axis=1)
test_dataset.drop(["Name","Ticket","Fare","Cabin"],inplace = True,axis=1)
print("Dropping Unwanted Columns -Name,Ticket,Fare,Cabin")

#Remove Invalid Data
cols = ["Age","SibSp","Parch"]
print("Started Removing the Invalid data from -{0}",cols )
for x in cols:
    train_dataset[x].fillna(train_dataset[x].mean(),inplace = True)
    test_dataset[x].fillna(test_dataset[x].mean(),inplace = True)

print("Successfully Removed the Invalid data from -{0}",cols )

#Convert the Categorical Data to Numerical Data
label_encoding = prepo.LabelEncoder()
columns = ["Sex","Embarked"]

print("Number of Columns to be Label Encoded is -{0} ",len(columns) )
for col in columns:
    train_dataset[col] = label_encoding.fit_transform(train_dataset[col])
    test_dataset[col] = label_encoding.fit_transform(test_dataset[col])
print("Successfully Encoded {0} columns  ",len(columns))

y=train_dataset["Survived"]
X=train_dataset.drop("Survived", axis =1)
print("Train Dataset started")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Successfully split the Train & Test Dataset of count {0} , {1} ",len(X_train),len(X_test))
#Creating an LogisticRegression
clf = LogReg(random_state=0, max_iter=1000).fit(X_train,y_train)
prediction_log =clf.predict(X_test)
submission_preds = clf.predict(test_dataset)
print(accuracy_score(y_test,prediction_log))
dataframe = pd.DataFrame({"PassengerId" : test_Passenger_Ids.values  ,"Survived": submission_preds ,})
dataframe.to_csv("Submission_Logistic.csv",index=False)

#Creating an LinearRegression
regresson=LinReg()  
regresson.fit(X_train,y_train)  

#To retrieve the intercept:
print(regresson.intercept_)

#For retrieving the slope:
print(regresson.coef_)
#preidction_reg = regresson.predict(X_test) 
y_pred = regresson.predict(X_test)
##print(accuracy_score(y_test,preidction_reg))
# setting plot style

plt.style.use('fivethirtyeight')

  
## plotting residual errors in training data
plt.scatter(regresson.predict(X_train), regresson.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  
## plotting residual errors in test data
plt.scatter(regresson.predict(X_test), regresson.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plot title
plt.title("Titanic Survivors")
## plotting legend
plt.legend(loc = 'upper right')
plt.show()


  
## method call for showing the plot

dataframe = pd.DataFrame({"PassengerId" : y_test.values  ,"Survived": y_pred ,})
##Download the CSV
dataframe.to_csv("Submission_Linear.csv",index= False)
