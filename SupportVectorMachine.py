import sklearn.datasets as data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class SupportVectorMachine(object):
   
    def SupportVectorMachineModel(self):
         print("Fetching all Data form Load Digits Datasets")
         iris = data.load_iris()
         data_frame = pd.DataFrame(iris.data,columns=iris.feature_names)
         print("Generating new column Flower name based on Index")
         data_frame['target'] = iris.target
         data_frame['Flower_Name'] = data_frame.target.apply(lambda x:iris.target_names[x])
         print(data_frame)
         data_frame0 = data_frame[data_frame.target == 0]
         data_frame1 = data_frame[data_frame.target == 1]
         data_frame2 = data_frame[data_frame.target == 2]
         print('Sepal Graph')
         plt.xlabel('sepal length (cm)')
         plt.ylabel('sepal width (cm)')
         plt.scatter(data_frame0['sepal length (cm)'],data_frame0['sepal width (cm)'],color='green',marker='*')
         plt.scatter(data_frame1['sepal length (cm)'],data_frame1['sepal width (cm)'],color='blue',marker='+')
         plt.show()
         print('Petal Graph')
         plt.xlabel('petal length (cm)')
         plt.ylabel('petal width (cm)')
         plt.scatter(data_frame0['petal length (cm)'],data_frame0['petal width (cm)'],color='green',marker='*')
         plt.scatter(data_frame1['petal length (cm)'],data_frame1['petal width (cm)'],color='blue',marker='+')
         plt.show()
         print("Removing Unwanted Columns")
         X = data_frame.drop(['target','Flower_Name'],axis=1)
         y = data_frame.target
         X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
         model = SVC()
         model.fit(X_train,y_train)
         print(model.score(X_test,y_test))

    def  SupportVectorMachineModel_Excercise(self):
          print("Fetching all Data form Load Digits Datasets")
          digits = data.load_digits() 
          data_frame = pd.DataFrame(digits.data,digits.target)
          data_frame['target'] = digits.target
          X_train,X_test,y_train,y_test = train_test_split(data_frame.drop('target',axis=1),data_frame['target'],test_size=0.2)
          model = SVC(kernel='linear',gamma=10)
          model.fit(X_train,y_train)
          print("When the model kernal is Linear .The accuracy is {}".format(model.score(X_test,y_test)))
          model = SVC(kernel='rbf')
          model.fit(X_train,y_train)
          print("When the model kernal is rbf .The accuracy is {}".format(model.score(X_test,y_test)))

svm = SupportVectorMachine()
svm.SupportVectorMachineModel_Excercise()

 