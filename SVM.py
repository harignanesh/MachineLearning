import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

print("Get Data from sklearn datasets")
cancer = load_breast_cancer()

df_feature = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

print(df_feature.info())
X = df_feature
y = cancer['target']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=44)

svc = SVC()
svc.fit(X_train,y_train)
predictions_svc = svc.predict(X_test)

print(confusion_matrix(predictions_svc,y_test))
print("\n")
print(classification_report(predictions_svc,y_test))

param_grid ={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,verbose=5)

grid.fit(X_train,y_train)

grid_prediction = grid.predict(X_test)
print(confusion_matrix(grid_prediction,y_test))
print("\n")
print(classification_report(grid_prediction,y_test))
