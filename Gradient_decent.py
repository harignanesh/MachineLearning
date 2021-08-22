import pandas as pd
import numpy as np
import math as m
import sklearn.linear_model as sklin
#Read_csv
print("Read the CSV File")
subject_marks = pd.read_csv("D:/ML/py-Master/py-master/ML/3_gradient_descent/Exercise/test_scores.csv")
print("Successfully Read the CSV Files")
x_array = np.array(subject_marks.math.to_list())
y_array = np.array(subject_marks.cs.to_list())


def gradient_decent(x,y):
    m_curr = b_curr = 0
    iterations = 121
    n = len(x)
    learning_rate = 0.002
    cost_previous =0
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([var** 2 for var in (y - y_predicted)])
    
        md = -(2 / n) * sum(x*(y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if m.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print("The m_curr is {} and the b_curr is {} which has cost function {} with the iterations {}".format(m_curr,b_curr,cost,i))

def Linear_regression(x,y):
    print("Read the CSV File")
    subject_marks = pd.read_csv("D:/ML/py-Master/py-master/ML/3_gradient_descent/Exercise/test_scores.csv")
    print("Successfully Read the CSV Files")
    reg = sklin.LinearRegression()
    reg.fit(subject_marks[['math']],subject_marks.cs)
    print("The Co-efficient is {}".format(reg.coef_))
    print("The Y-intercept is {}".format(reg.intercept_))


gradient_decent(x_array,y_array)
Linear_regression(x_array,y_array)


