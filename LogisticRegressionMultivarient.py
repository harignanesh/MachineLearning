import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sb
import sklearn.datasets as data
class LogisticRegressionMultivarient(object):

    
   def Digits_finder():
        print("Fetching all Data form Load Digits Datasets")
        digits = load_digits()
  
        print("Split the data for Training & Testing")
        X_train,X_test,y_train,y_test =train_test_split(digits.data, digits.target,test_size=0.2)

        print("Create an Instance of LogisticRegression")
        LogReg = LogisticRegression()
        print("Training the Model")
        LogReg.fit(X_train,y_train)
        y_Predecited = LogReg.predict(X_test)
        print("The Accuracy of the Model is {}".format(LogReg.score(X_test,y_test)))
        
        print("The Confusion Matrix")
        cm = confusion_matrix(y_test,y_Predecited)
        plt.figure(figsize=(10,7))
        sb.heatmap(cm,annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

   def Flower_detector(self):
         print("Fetching all Data form Load Digits Datasets")
         iris = data.load_iris()
         print("Split the data for Training & Testing")
         X_train,X_test,y_train,y_test =train_test_split(iris.data, iris.target,test_size=0.2)
         print("Create an Instance of LogisticRegression")
         LogReg = LogisticRegression()
         print("Training the Model")
         LogReg.fit(X_train,y_train)
         y_Predecited = LogReg.predict(X_test)
         print("The Accuracy of the Model is {}".format(LogReg.score(X_test,y_test)))
         print("The Confusion Matrix")
         cm = confusion_matrix(y_test,y_Predecited)
         plt.figure(figsize=(10,7))
         sb.heatmap(cm,annot=True)
         plt.xlabel('Predicted')
         plt.ylabel('Truth')
         plt.show()

LogisticMultiVar = LogisticRegressionMultivarient()
LogisticMultiVar.Flower_detector()