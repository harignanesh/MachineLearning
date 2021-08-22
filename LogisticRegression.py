import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Read the CSV File
class LogisticRegressionTraining:

  def LogReg_Main():
      print("Read the CSV File")
      insurance_data=pd.read_csv('D:/ML/py-Master/py-master/ML/7_logistic_reg/insurance_data.csv')
      print("Successfully Read the CSV Files")
      #plt.scatter(insurance_data.age,insurance_data.bought_insurance,marker="+",color="red")
      #plt.show()
      print("Splitting the data to Train and Test")
      X_Train,X_Test,y_Train,y_Test = train_test_split(insurance_data[['age']],insurance_data.bought_insurance,test_size=0.1)
      print("Creating an Instance of the LogisticRegression")
      LogReg = LogisticRegression()
      print("Training Model")
      LogReg.fit(X_Train,y_Train)
      print("Predecting Test DataSet for that Model")
      print(LogReg.predict(X_Test))
      print("Getting the Accuracy of that Model")
      print(LogReg.score(X_Test,y_Test))

  def LogReg_Excercise(self):
      print("Read the CSV File")
      HR_comma_sep=pd.read_csv('D:/ML/py-Master/py-master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv')
      print("Successfully Read the CSV Files")
      print("Applying onehotencoding")
      dummy= pd.get_dummies(HR_comma_sep['salary'])
      print("Appending the Dummy with main dataset")
      merged = pd.concat([HR_comma_sep,dummy],axis=1)
      print("Dropping the Unwanted columns")
      HR_comma_sep = merged.drop(['salary','low'],axis='columns')
     
      print("The Final Dataset")
      HR_comma_sep_x = HR_comma_sep.drop(['last_evaluation','number_project','Work_accident','Department','last_evaluation','left'],axis=1,inplace= False)
      HR_comma_sep_y=HR_comma_sep.left
      print(HR_comma_sep.info())
      print(HR_comma_sep_x.info())
      print("Splitting the data to Train and Test")
      X_Train,X_Test,y_Train,y_Test = train_test_split(HR_comma_sep_x,HR_comma_sep_y,test_size=0.1,random_state = 45)
      print("Creating an Instance of the LogisticRegression")
      LogReg = LogisticRegression()
      print("Training Model")
      LogReg.fit(X_Train,y_Train)
      print("Predecting Test DataSet for that Model")
      print(LogReg.predict(X_Test))
      print("Getting the Accuracy of that Model")
      print(LogReg.score(X_Test,y_Test))

l = LogisticRegressionTraining()
l.LogReg_Excercise()