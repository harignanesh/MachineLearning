import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from word2number import w2n
import sklearn.externals as ske

class practice_oneHotencoding:

    def oneHotencoding(self):
     print("Read the CSV File")
     homeprices = pd.read_csv("D:/ML/py-Master/py-master/ML/5_one_hot_encoding/homeprices.csv")
     print("Successfully Read the CSV Files")
     
     print("Applying onehotencoding")
     dummy= pd.get_dummies(homeprices['town'])
     print(dummy)
     print("Appending the Dummy with main dataset")
     merged = pd.concat([homeprices,dummy],axis=1)

     print("Dropping the Unwanted columns")
     finalset = merged.drop(['town','robinsville'],axis='columns')
     
     print("The Final Dataset")
     print(finalset)

     print("Creating an Instance of Linear model")
     reg = linear_model.LinearRegression()
     print("Training the Model")
     X = finalset.drop(['price'],axis=1)
     y = finalset.price
     reg.fit(X,y)
     
     print("The Co-Efficent is :{0}",reg.coef_)
     print("The Intercept is :{0}",reg.intercept_)
     print("The Accuracy of the model is {}".format(reg.score(X,y)*100))
     print("Predecting the value for 2500sq.ft , 4 bedrooms, 5 years of age : {0}",reg.predict([[2500,4,5]]))
    
    def oneHotencoding_Excercise(self):
     print("Read the CSV File")
     carprices = pd.read_csv("D:/ML/py-Master/py-master/ML/5_one_hot_encoding/Exercise/carprices.csv")
     print("Successfully Read the CSV Files")
     carprices.columns = carprices.columns.str.replace(' ','_')
     print("Applying onehotencoding")
     dummy= pd.get_dummies(carprices['Car_Model'])
     print("Appending the Dummy with main dataset")
     merged = pd.concat([carprices,dummy],axis=1)
     print("Dropping the Unwanted columns")
     finalset = merged.drop(['Mercedez Benz C class','Car_Model'],axis='columns')
     
     print("The Final Dataset")

     print("Creating an Instance of Linear model")
     reg = linear_model.LinearRegression()
     print("Training the Model")
     X = finalset.drop(['Sell_Price'],axis=1)
     y = finalset.Sell_Price
     reg.fit(X,y)
     print(finalset)
     print("The Co-Efficent is :{0}",reg.coef_)
     print("The Intercept is :{0}",reg.intercept_)
     print("The Accuracy of the model is {}".format(reg.score(X,y)*100))
     print("Predecting the value for mercedec which is 4 yeara and 45000 milage : {0}",reg.predict([[45000,4,0,0]]))
     print("Predecting the value for BMW X5 which is 7 yeara and 86000 milage : {0}",reg.predict([[86000,7,0,1]]))
    
p =practice_oneHotencoding()
p.oneHotencoding_Excercise();