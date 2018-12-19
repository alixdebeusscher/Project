from tools import *
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def normalize(X_train):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    #x_train = pd.DataFrame(x_train_scaled)
 
    return x_train_scaled

def get_k(X_train,X_test,y_train,y_test,max_k):
    rmse_val = [] #to store rmse values for different k
    rmse_train = []
    for K in range(1,max_k+1):
        model = neighbors.KNeighborsRegressor(n_neighbors = K)
        model.fit(X_train, y_train)  #fit the model
        predt=model.predict(X_train)
        errort=sqrt(mean_squared_error(y_train,predt))
        pred=model.predict(X_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
        rmse_train.append(errort)
        #print('RMSE value for k= ' , K , 'is:', error)
        
    #PLOT K
    fig = plt.figure()

    plt.plot(range(1,max_k+1), rmse_val, label='Test set')
    plt.plot(range(1,max_k+1), rmse_train, label= 'Training set')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.show()

#    curve = pd.DataFrame(rmse_val) #elbow curve 
#    curvet = pd.DataFrame(rmse_train)
#    curve.plot()
#    curvet.plot()
    k = min(rmse_val)
    p = rmse_val.index(min(rmse_val))+1
    return k,p

def linear(X,y):
    regressor = LinearRegression()
      
    regressor.fit(X, y)  
    print(regressor.coef_)
    
    width = 1/1.5
    plt.figure(figsize=(10,3))
    plt.bar(col[:-1], regressor.coef_, width, color="green")
    plt.savefig('coef.png')
    
    plot_learning_curve(LinearRegression(), '', X, y, cv=5).savefig('lr.png')

#Read test file
X1=pd.read_csv("X1_t1.csv")

#column name
col = list(X1)

#Split test file in learning set and test set
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values


linear(X,y)
X_scaled = normalize(X)
linear(X_scaled,y)
#split data
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, shuffle=False)
#rmse, k = get_k(X_train,X_test,y_train,y_test,100)
#print('Least rmse is:', rmse, 'with k:', k)
#print(X_train[4])
##normalize data
#x_train = normalize(X_train)
#x_test = normalize(X_test)
#print(x_train[4])
#rmse, k = get_k(x_train,x_test,y_train,y_test,100)
#print('Least rmse is:', rmse, 'with k:', k)


