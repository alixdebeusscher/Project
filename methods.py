import tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error 
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import itertools as it

def data_analysis(X,X_n,y,col):
    #PLot the target VS each features (normalized)
    plt.figure(figsize=(14,6))
    for i in range(8):
        plt.subplot(3,3,i+1)
        plt.scatter(X_n[:,i],y,s=10)
        plt.title(col[i])   
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.40,
                    wspace=0.1)
    plt.savefig('scatterFeatures.png')
    
    #function used for the estimation of mutual information
    def distribution(X):
        dim = X.ndim
        size = X.shape
        count = {}
        if dim == 1:
            for i in range(size[0]):
                entry = X[i]
                if entry not in count:
                    count[entry] = 0
                count[entry] += 1
        else:           
            for i in range(size[0]):
                entry = tuple(X[i,:])
                if entry not in count:
                    count[entry] = 0
                count[entry] += 1
        return {k:v/size[0] for (k,v) in count.items()}
    
    def entropy(distr):
        p = np.array([v for v in distr.values()])
        return - (p * np.log2(p)).sum()
    
    def concat(X1,X2):
        return np.concatenate(([X1],[X2]),axis=0).T
    
    # estimate the MI and do the plot
    corr = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            d12 = distribution(concat(X[:,i],X[:,j]))
            d1  = distribution(X[:,i])
            d2  = distribution(X[:,j])
            E   = entropy(d1) + entropy(d2)
            corr[i,j] = (E - entropy(d12)) / entropy(d12)
            
    sns.clustermap(corr, annot=True, xticklabels = col, yticklabels = col);
    plt.savefig('mutualInformation.png')
    
    #estimate the MI of each features with the target
    corry = np.zeros(8)
    for i in range(8):
        d12 = distribution(concat(X[:,i],y))
        d1  = distribution(X[:,i])
        d2  = distribution(y)
        E   = entropy(d1) + entropy(d2)
        corry[i] = (E - entropy(d12)) / entropy(d12)   
    for i in range(8):
        print("MI of %s is: %f" % (col[i], corry[i]))


def linear(X,y,col):
    regressor = LinearRegression()
      
    regressor.fit(X, y)  
    print('Coefficient of the linear regression : ',regressor.coef_)
    
    width = 1/1.5
    plt.figure(figsize=(10,3))
    plt.bar(col, regressor.coef_, width, color="green")
    plt.savefig('coef.png')
    
    tools.plot_learning_curve(regressor, '', X, y, cv=5)
    plt.savefig('lr.png')
    
    
def kNN(X,y,col):
    # Used for the plot with several k
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
        #PLOT K
        plt.figure()
        plt.plot(range(1,max_k+1), rmse_val, label='Test set')
        plt.plot(range(1,max_k+1), rmse_train, label= 'Training set')
        plt.legend(loc = "best")
        plt.xlabel('Value of k (number of Neighbors)')
        plt.ylabel('RMSE')
        plt.savefig('valuesk.png')
        plt.show()
    
        k = min(rmse_val)
        p = rmse_val.index(min(rmse_val))+1
        return k,p
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, shuffle=False)
    rmse, k = get_k(X_train,X_test,y_train,y_test,100)
    print('Least rmse is:', rmse, 'with k:', k)
    #normalize data
    x_train = tools.normalize(X_train)
    x_test = tools.normalize(X_test)
    rmse, k = get_k(x_train,x_test,y_train,y_test,100)
    print('Least rmse is:', rmse, 'with k:', k)
    
    # Use to fit the best k and  eatures to the kNN
    def best_features_meta_parameter(X,y,estimator,param_grid):
        size = X.shape
        minscore = float('Inf')
        FinalScores = None
        bestpar = None
        bestfeatures = None
        for k in param_grid:
            model = estimator(n_neighbors=k)
            for n_features in range(1,size[1]+1):
                for features in it.combinations(list(range(size[1])), n_features):
                    subdata = X[:,features]
                    scores = cross_val_score(model,subdata,y, cv=10, scoring=tools.score_function)
                    if scores.mean() < minscore:
                        minscore = scores.mean()
                        FinalScores = scores
                        bestpar = k
                        bestfeatures = features
        return FinalScores,bestpar,bestfeatures
    
    #param_grid = {'n_neighbors':[2, 4, 6, 8, 10]}
    #model = GridSearchCV(neighbors.KNeighborsRegressor, param_grid, scoring='neg_mean_absolute_error', cv=10)
    #model.fit(X,y)
    estimator = neighbors.KNeighborsRegressor
    (Result,k,features) = best_features_meta_parameter(X,y,estimator,param_grid = range(1,5))
    print('BEST MODEL FOR KNN')
    print('k = ',k)
    print('features = ',features)
    model = estimator(n_neighbors=k)
    tools.plot_learning_curve(model, 'Test', X[:,features], y, cv=10)
