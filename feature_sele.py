import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as mp
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
def chooseWrapper(X1):
    #Split test file in learning set and test set
    X = X1.drop('ccs',axis=1).values
    y = X1['ccs'].values
    for each in colom:
        print(each)
        X=X1['cement'].values
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, shuffle=False)

        #D'autres moyen de normaliser ?????????? je comprends pas trop cette section
#        scaler = MinMaxScaler(feature_range=(0, 1))
#        
#        x_train_scaled = scaler.fit_transform(X_train)
#        x_train = pd.DataFrame(x_train_scaled)
#        
#        x_test_scaled = scaler.fit_transform(X_test)
#        x_test = pd.DataFrame(x_test_scaled)
        
            
        rmse_val = [] #to store rmse values for different k
        for K in range(22):
            K = K+1
            model = neighbors.KNeighborsRegressor(n_neighbors = K)
            model.fit(x_train, y_train)  #fit the model
            pred=model.predict(x_test) #make prediction on test set
            error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
            rmse_val.append(error) #store rmse values
            #print('RMSE value for k= ' , K , 'is:', error)
            
        
            k = min(rmse_val)
            print(each,k, K)
        
    #2
    #3
    #4
    #5
    #6
    #7
    
#Read test file
X1=pd.read_csv("X1_t1.csv")
chooseWrapper(X1)

