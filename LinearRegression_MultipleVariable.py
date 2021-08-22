import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
from word2number import w2n
import sklearn.externals as ske

class practice:

    def home_price_calculator(self):
     print("Read the CSV File")
     homeprices = pd.read_csv("D:/ML/py-Master/py-master/ML/2_linear_reg_multivariate/homeprices.csv")
     print("Successfully Read the CSV Files")
     
     print("Removing the unwanted values")
     homeprices_bedroom_nan = math.floor(homeprices.bedrooms.median())
     homeprices.bedrooms = homeprices.bedrooms.fillna(homeprices_bedroom_nan)
     print("Successfully Removed the unwanted values")
     
     print("Creating an Instance of Linear model")
     reg = linear_model.LinearRegression()
     print("Training the Model")
     reg.fit(homeprices[['area','bedrooms','age']],homeprices.price)
     
     print("The Co-Efficent is :{0}",reg.coef_)
     print("The Intercept is :{0}",reg.intercept_)
     
     print("Predecting the value for 2500sq.ft , 4 bedrooms, 5 years of age : {0}",reg.predict([[2500,4,5]]))
     ske.joblib.dump(homeprices,'Linear_Reg_HomepRice')
     ske.joblib.load()

    def Calculate_Salary_onExperiance(self):
     print("Read the CSV File")
     hiring = pd.read_csv("D:/ML/py-Master/py-master/ML/2_linear_reg_multivariate/Exercise/hiring.csv")
     print("Successfully Read the CSV Files")
     
     print("Refactoring the Data")
     hiring.columns = hiring.columns.str.replace(' ','_')
     hiring =hiring.rename(columns={'test_score(out_of_10)':'test_score'})
     hiring =hiring.rename(columns={'interview_score(out_of_10)':'interview_score'})
     hiring =hiring.rename(columns={'salary($)':'salary'})
     print("Filling Zero value in case of Nan for Experience")
     hiring.experience = hiring.experience.fillna('zero')
     print("Filling median value in case of Nan for Test_score")
     test_score_nan = hiring.test_score.median()
     hiring.test_score = hiring.test_score.fillna(test_score_nan)
     print("Converting the Words to Number for Experience")
     hiring.experience = hiring.experience.apply(w2n.word_to_num)
     print("Successfully Removed the unwanted values")
     print("Creating an Instance of Linear model")
     reg = linear_model.LinearRegression()
     print("Training the Model")
     reg.fit(hiring[['experience','test_score','interview_score']],hiring.salary)
     print("The Co-Efficent is :{0}",reg.coef_)
     print("The Intercept is :{0}",reg.intercept_)
     print(reg.predict([[2,9,6]]))    
     print(reg.predict([[12,10,10]]))






pract = practice()
pract.home_price_calculator()
