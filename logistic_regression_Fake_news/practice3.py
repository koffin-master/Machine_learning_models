import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.stats import alpha


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6)
matplotlib.rcParams['figure.facecolor'] = 'white'

medical_df = pd.read_csv("/Users/rahmani/Documents/Assets_ML/expenses.csv")

non_smoker_df = medical_df[medical_df.smoker =="no"]
ages = non_smoker_df.age
w = 50
b = 100

def estimate_charges(age, W_age,bmi, W_bmi,children ,W_children , b):
    return W_age * age + W_bmi * bmi + W_children * children + b

# estimated_charges = estimate_charges(ages,w,b)
# print(estimated_charges)

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))

def try_parameter(W_age, W_bmi,W_children, b):
    ages = non_smoker_df.age
    bmis = non_smoker_df.bmi
    child = non_smoker_df.children
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, W_age,bmis ,W_bmi,child,W_children,  b)
    plt.plot(ages, predictions, 'r',alpha=0.9)
    plt.scatter(ages,target,s=8,alpha=0.8)
    plt.scatter(bmis,target,c='g')
    plt.scatter(child,target,c='orange')
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.legend(['Estimate','Actual'])
    loss = rmse(target, predictions)
    print("RMSE Loss", loss)
    plt.show()

# Create inputs and targets
inputs, targets = non_smoker_df[['age', 'bmi','children']], non_smoker_df['charges']

# Create and train model
model = LinearRegression()
model.fit(inputs, targets)
predictions = model.predict(inputs)
print("W is\n",model.coef_,"\n")
print("b is\n",model.intercept_,"\n")
print("RMSE :", rmse(predictions,targets))

try_parameter(model.coef_[0] ,model.coef_[1],model.coef_[2], model.intercept_)