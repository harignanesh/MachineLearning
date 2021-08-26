import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
class DecisionTree(object):
    
   def Finding_Salary_MoreThan10k(self):
        print("Fetching all Data")
        GreatherSalary = pd.read_csv("D:/ML/py-Master/py-master/ML/9_decision_tree/salaries.csv")
        print("Successfully Read all Data")
        inputs = GreatherSalary.drop("salary_more_then_100k",axis=1)
        target=GreatherSalary["salary_more_then_100k"]
        print("Encoding the datas")
        le_company =LabelEncoder()
        le_job = LabelEncoder()
        le_degree = LabelEncoder()
        inputs["company_n"] = le_company.fit_transform(inputs["company"])
        inputs["job_n"] = le_job.fit_transform(inputs["job"])
        inputs["degree_n"] = le_degree.fit_transform(inputs["degree"])
        print(inputs.head(5))
        print("Dropping the Unwanted Columns")
        inputs_n =inputs.drop(['company','job','degree'],axis=1)
        print(inputs_n.head(5))
        print("Split the data for Training & Testing")
        X_train,X_test,y_train,y_test =train_test_split(inputs_n, target,test_size=0.2)
     
        print("Create an Instance of LogisticRegression")
        D_Tree = DecisionTreeClassifier()
        print("Training the Model")
        D_Tree.fit(X_train,y_train)
        y_Predecited = D_Tree.predict(X_test)
        print("The Accuracy of the Model is {}".format(D_Tree.score(X_test,y_test)))
        

dt = DecisionTree()
dt.Finding_Salary_MoreThan10k()
