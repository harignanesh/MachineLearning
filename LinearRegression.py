import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
#Read the CSV File

cpc_income_dataset=pd.read_csv('D:/ML/py-Master/py-master/ML/1_linear_reg/Exercise/canada_per_capita_income.csv')
plt.xlabel("year")
plt.ylabel("Per_captia_Income")
cpc_income_dataset.columns = cpc_income_dataset.columns.str.replace(' ','_')
plt.scatter(cpc_income_dataset.year,cpc_income_dataset.per_capita_income,color='Red',marker="o")

regression = linear_model.LinearRegression()
regression.fit(cpc_income_dataset[['year']],cpc_income_dataset.per_capita_income)
plt.plot(cpc_income_dataset.per_capita_income,regression.predict(cpc_income_dataset[['year']]))


#regression coef
print("Co-efficent : {0} ",regression.coef_)
#regression_intercept
print("Y- Intercept : {0} ",regression.intercept_)
#regreesion 
print(regression.predict([['2020']]))

#cpc_dataset=pd.read_csv('D:/ML/py-Master/py-master/ML/1_linear_reg/Exercise/canada_per_capita_income.csv')
#cpc_dataset.head(3)
#p = regression.predict(cpc_dataset[['year']])

#cpc_dataset['Forecasted-year'] = p

#print(cpc_dataset,index=False)