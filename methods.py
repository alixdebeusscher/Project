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
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

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
    
def mlp_solver_activation(X_n,y,col):
    mlp = MLPRegressor(max_iter=2000) 
    parameter_space = {
        'solver': ['sgd', 'adam', 'lbfgs'],
        'activation' : ['identity', 'logistic', 'tanh', 'relu']
    }
    
    # do stuff
    clf = GridSearchCV(mlp, parameter_space, cv=5, scoring = tools.score_function_neg)
    clf.fit(X_n, y)

    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)
    
    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
def mlp_nbr_neurons(X_n,y,col):
    mlp = MLPRegressor(max_iter=2000,solver='lbfgs',activation='relu')

    train_sizes = range(1,31)
    NumberNeurons1 = []
    NumberNeurons2 = []
    NumberNeurons3 = []
    for i in train_sizes:
        NumberNeurons1.append((i,))
        NumberNeurons2.append((i,)*3)
        NumberNeurons3.append((i,)*5)
        
    parameter_space1 = {
        'hidden_layer_sizes': NumberNeurons1,
    }
    parameter_space2 = {
        'hidden_layer_sizes': NumberNeurons2,
    }
    parameter_space3 = {
        'hidden_layer_sizes': NumberNeurons3,
    }
    
    clfNbrN1 = GridSearchCV(mlp, parameter_space1, cv=5, scoring = tools.score_function_neg)
    clfNbrN2 = GridSearchCV(mlp, parameter_space2, cv=5, scoring = tools.score_function_neg)
    clfNbrN3 = GridSearchCV(mlp, parameter_space3, cv=5, scoring = tools.score_function_neg)
    clfNbrN1.fit(X_n, y)
    clfNbrN2.fit(X_n, y)
    clfNbrN3.fit(X_n, y)   
    
    # Best paramete set
    print('Best parameters found:\n', clfNbrN1.best_params_)
    print('Best parameters found:\n', clfNbrN2.best_params_)
    print('Best parameters found:\n', clfNbrN3.best_params_)
    
    # All results
    means1 = -clfNbrN1.cv_results_['mean_test_score']
    stds1 = -clfNbrN1.cv_results_['std_test_score']
    means2 = -clfNbrN2.cv_results_['mean_test_score']
    stds2 = -clfNbrN2.cv_results_['std_test_score']
    means3 = -clfNbrN3.cv_results_['mean_test_score']
    stds3 = -clfNbrN3.cv_results_['std_test_score']
    
    # PLot
    plt.figure()
    plt.xlabel("Number of neurones")
    plt.ylabel("RMSE")
    plt.grid()
    plt.fill_between(train_sizes, means1 - stds1,
                     means1 + stds1, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, means2 - stds2,
                     means2 + stds2, alpha=0.1, color="g")
    plt.fill_between(train_sizes, means3 - stds3,
                     means1 + stds3, alpha=0.1,
                     color="b")
    plt.plot(train_sizes, means1, color="r",
             label="One layer")
    plt.plot(train_sizes, means2, color="g",
             label="Three layers")
    plt.plot(train_sizes, means3, color="b",
             label="Five layers")
    plt.legend(loc="best")
    plt.savefig('NbrNeurones.png')
         
def mlp_nbr_layers(X_n,y,col):
    mlp = MLPRegressor(max_iter=2000,solver='lbfgs',activation='relu')
    
    NumberLayers1 = []
    NumberLayers2 = []
    NumberLayers3 = []
    train_sizes = range(1,11)
    for i in train_sizes:
        NumberLayers1.append((5,)*i)
        NumberLayers2.append((15,)*i)
        NumberLayers3.append((30,)*i)
        
    parameter_space1 = {
        'hidden_layer_sizes': NumberLayers1,
    }
    parameter_space2 = {
        'hidden_layer_sizes': NumberLayers2,
    }
    parameter_space3 = {
        'hidden_layer_sizes': NumberLayers3,
    }
    
    clfNbrL1 = GridSearchCV(mlp, parameter_space1, cv=5, scoring = tools.score_function_neg)
    clfNbrL2 = GridSearchCV(mlp, parameter_space2, cv=5, scoring = tools.score_function_neg)
    clfNbrL3 = GridSearchCV(mlp, parameter_space3, cv=5, scoring = tools.score_function_neg)
    clfNbrL1.fit(X_n, y)
    clfNbrL2.fit(X_n, y)
    clfNbrL3.fit(X_n, y)
    
    # Best paramete set
    print('Best parameters found:\n', clfNbrL1.best_params_)
    print('Best parameters found:\n', clfNbrL2.best_params_)
    print('Best parameters found:\n', clfNbrL3.best_params_)
    
    # All results
    means1 = -clfNbrL1.cv_results_['mean_test_score']
    stds1 = -clfNbrL1.cv_results_['std_test_score']
    means2 = -clfNbrL2.cv_results_['mean_test_score']
    stds2 = -clfNbrL2.cv_results_['std_test_score']
    means3 = -clfNbrL3.cv_results_['mean_test_score']
    stds3 = -clfNbrL3.cv_results_['std_test_score']
    
    # PLot
    plt.figure()
    plt.xlabel("Number of layers")
    plt.ylabel("RMSE")
    plt.grid()
    plt.fill_between(train_sizes, means1 - stds1,
                     means1 + stds1, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, means2 - stds2,
                     means2 + stds2, alpha=0.1, color="g")
    plt.fill_between(train_sizes, means3 - stds3,
                     means1 + stds3, alpha=0.1,
                     color="b")
    plt.plot(train_sizes, means1, color="r",
             label="Five neurones")
    plt.plot(train_sizes, means2, color="g",
             label="Fifteen neurones")
    plt.plot(train_sizes, means3, color="b",
             label="Thirty neurones")
    plt.legend(loc="best")
    plt.savefig('NbrLayers.png')
    
def mlp_final(X_n, y, solver, activation, n_neur_min, n_neur_max, n_layer_min, n_layer_max):
    mlp = MLPRegressor(max_iter=5000)
    
    StructNetwork = []
    for i in range(n_neur_min,n_neur_max+1):
        for j in range(n_layer_min,n_layer_max+1):
            StructNetwork.append((i,)*j)
        
    parameter_space = {
        'hidden_layer_sizes': StructNetwork,
        'solver':solver,
        'activation' : activation
    }
    
    clfNbrF = GridSearchCV(mlp, parameter_space, cv=5, scoring = tools.score_function_neg)
    clfNbrF.fit(X_n, y)
    
    # Best paramete set
    print('Best parameters found:\n', clfNbrF.best_params_)
    
    # PLot
    model = MLPRegressor(max_iter=2000,activation = 'tanh', 
                     hidden_layer_sizes = (13,),solver = 'lbfgs')
    tools.plot_learning_curve(model, '', X_n, y, cv=5)
    plt.savefig('FinalMLP.png')

def mlp_select_features(X_n,y,col,solver, activation, n_neur_min, n_neur_max, n_layer_min, n_layer_max)):
    mlpFeatures = MLPRegressor(max_iter=5000)
    
    StructNetwork = []
    for i in range(n_neur_min,n_neur_max+1):
        for j in range(n_layer_min,n_layer_max+1):
            StructNetwork.append((i,)*j)
        
    parameter_space = {
        'hidden_layer_sizes': StructNetwork,
        'solver':solver,
        'activation' : activation
    }
    
    #Apply the greedy algorithm to select the best features
    clfFeatures = GridSearchCV(mlpFeatures, parameter_space, cv=5, scoring = tools.score_function_neg)
    X_best = X_n
    nbrFeatures = 7
    toRemove = None
    BestParams = None
    NotBest = True 
    newCol = col
    clfFeatures.fit(X_n, y)
    minRMS = -clfFeatures.best_score_
    print(minRMS)
    print(clfFeatures.best_params_)
    while (NotBest):
        for i in range(1,nbrFeatures):
            X_try = np.delete(X_best,i,1)
            clfFeatures.fit(X_try, y)
            print(-clfFeatures.best_score_)
            if -clfFeatures.best_score_ <= minRMS:
                minRMS = -clfFeatures.best_score_
                toRemove = i
                print(toRemove)
                BestParams = clfFeatures.best_params_
                print(BestParams)
        if toRemove == None:
            NotBest = False
        else:
            newCol = np.delete(newCol,toRemove,0)
            X_best = np.delete(X_best,toRemove,1)
            nbrFeatures -= 1
            toRemove = None      
    
    print('FINAL SOLUTION')
    print(minRMS)
    print(BestParams)
    print(newCol)