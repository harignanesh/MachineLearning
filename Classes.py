import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv("D:/ML/titanic/gender_submission.csv")

passenger_surviced = []
passenger_deceased = []
survived_count = 0
Deceased_count = 0

for x in train_data.Survived:
    if x == 1:
       passenger_surviced.append(x)
    else:
     passenger_deceased.append(x)

print(len(passenger_surviced))
print(len(passenger_deceased))
constants =["Survived","Deceseased"]
values=[len(passenger_surviced),len(passenger_deceased)]
plt.bar(constants,values)
plt.show()