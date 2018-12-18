from tools import *
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def normalize(X_train):
    scaler = MinMaxScaler()

    x_train_scaled = scaler.fit_transform(X_train)
    x_train = pd.DataFrame(x_train_scaled)
    
    
    return x_train

def get_k(X_train,X_test,y_train,y_test,max_k):
    rmse_val = [] #to store rmse values for different k
    for K in range(1,max_k+1):
        model = neighbors.KNeighborsRegressor(n_neighbors = K)
        model.fit(x_train, y_train)  #fit the model
        pred=model.predict(x_test) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
        #print('RMSE value for k= ' , K , 'is:', error)
        
    #PLOT K
    curve = pd.DataFrame(rmse_val) #elbow curve 
    curve.plot()
    k = min(rmse_val)
    p = rmse_val.index(min(rmse_val))
    return k,p

def linear(X,y):
    regressor = LinearRegression()
      
    regressor.fit(X, y)  
    print(regressor.coef_)
    
    width = 1/1.5
    plt.figure(figsize=(10,3))
    plt.bar(col[:-1], regressor.coef_, width, color="green")
    plt.savefig('coef.png')
    
    plot_learning_curve(LinearRegression(), 'Test', X, y, cv=5)
    
    
    
  #  fig = plt.gcf()

    
#Read test file
X1=pd.read_csv("X1_t1.csv")

#column name
col = list(X1)

#Split test file in learning set and test set
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values

X_scaled = normalize(X)
linear(X,y)
linear(X_scaled,y)
#split data
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, shuffle=True)

#normalize data
x_train, x_test = normalize(X_train, X_test)
rmse, k = get_k(X_train,X_test,y_train,y_test,30)
print('Least rmse is:', rmse, 'with k:', k)


