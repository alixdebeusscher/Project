import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls


dataset = pd.read_csv('X1_t1.csv') 
colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
dataset.shape
#print(dataset.head())

X = dataset.drop('ccs',axis=1).values
y = dataset['ccs'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  

regressor = LinearRegression()
  
regressor.fit(X_train, y_train)  
print('EIC', regressor.intercept_)
print('NOC', len(regressor.coef_))
y_pred = regressor.predict(X_test)  

k = 0
tab = [0, 0, 0, 0, 0, 0, 0, 0]
for each in colom:
    for i in range(10000):
        tab[k]+=regressor.coef_[k]/10000
    print("Coef of %s is: %f" %(each, tab[k]))        
    k+=1

y = tab
N = len(y)
x = range(N)
width = 1/1.5
plt.bar(x, y, width, color="blue")


fig = plt.gcf()

    