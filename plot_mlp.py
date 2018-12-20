import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tools 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def normalize(X_train):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    return x_train_scaled

def mlp_fun(Xtrain, Ytrain, solver, activation, n_neur_min, n_neur_max, n_layer_min, n_layer_max):
    mlp = MLPRegressor(max_iter=50000)
    neur=[]
    for i in range(n_neur_min,n_neur_max+1):
        for j in range(n_layer_min, n_layer_max+1):
            neur.append((i,)*j)
    parameter_space = {
        'hidden_layer_sizes': neur,
        'solver':solver,
        'activation' : activation
    }
    clf = GridSearchCV(mlp, parameter_space, cv=5, scoring=tools.score_function_neg)
    f = time.time()
    clf.fit(Xtrain, Ytrain)
    el = time.time() - f
    print(el)
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    result = []
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        result.append((-mean, std, params))
    return result

def plot_for_solver(solver, activation, result, n_neur_min, n_neur_max):
    plt.subplots(1)
    for i in range(0,len(activation)):
        toplot = []
        for j in range(0, n_neur_max-n_neur_min):
            mean = 0
            for k in range(0, len(solver)):
               mean+=result[i*(n_neur_max-n_neur_min)*len(solver)+j*len(solver)+k][0]
            mean=mean/len(solver)
            toplot.append(mean)
        plt.plot(toplot, label=activation[i])   
    plt.legend()
    plt.xlabel('Number of neurons')
    plt.ylabel('RMSE')
    name = "sol"+ str(n_neur_min)+'_'+ str(n_neur_max)
    plt.savefig(name)
 
def plot_for_activation(solver, activation, result, n_neur_min, n_neur_max):
    plt.subplots(1)
    for i in range(0,len(solver)):
        toplot = []
        for j in range(0, n_neur_max-n_neur_min):
            mean = 0
            for k in range(0, len(activation)):
               mean+=result[i+j*len(solver)+k*len(solver)*(n_neur_max-n_neur_min)][0]
            mean=mean/len(activation)
            toplot.append(mean)
        plt.plot(toplot, label=solver[i])   
    plt.legend()
    plt.xlabel('Number of neurons')
    plt.ylabel('RMSE')
    name = "act"+ str(n_neur_min)+'_'+ str(n_neur_max)
    plt.savefig(name)


X1=pd.read_csv("X1_t1.csv")
col = list(X1)
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values
Xtrain = normalize(X)
Ytrain = y

solver = ['lbfgs','sgd','adam']
activation = ['tanh','relu','identity','logistic']
n_neur_min = 10
n_neur_max = 20
n_layer_min = 1
n_layer_max = 3


result = mlp_fun(Xtrain, Ytrain, solver, activation, n_neur_min, n_neur_max)
print(result)
plot_for_solver(solver, activation, result, n_neur_min, n_neur_max)
plot_for_activation(solver, activation, result, n_neur_min, n_neur_max)


#%%Predict Y2 with best model
X1=pd.read_csv("X1_t1.csv")
col = list(X1)
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values
Xtrain = normalize(X)
X_train,X_test,y_train,y_test = train_test_split(Xtrain,y,test_size=0.20, shuffle=False)
#Ytrain = y

mlp = MLPRegressor(hidden_layer_sizes=(13,), activation='tanh', solver='lbfgs', max_iter=50000)
mlp.fit(Xtrain,y)
FileToPredict=pd.read_csv("X2.csv")
FileToPredict=normalize(FileToPredict)
pred = mlp.predict(FileToPredict)
print(pred)
print(y_test)


#pred = pd.DataFrame(pred)
#pred = pred.values


#%%
np.savetxt("Y2.csv", pred)
#%%
X1=pd.read_csv("X1_t1.csv")
X2=pd.read_csv("X2.csv")
col = list(X1)
X = X1.drop(col[-1],axis=1).values
X=pd.DataFrame(X)
Xpred = X2.values
Xpred=pd.DataFrame(Xpred)
y = X1[col[-1]].values
frames = [X, Xpred]
toNorm=pd.concat(frames)
toDiv=normalize(toNorm)
Xtrain = toDiv[:515]
X_pred = toDiv[-515:]
Ytrain = y

#%%Predict Y2 with best model
mlp = MLPRegressor(hidden_layer_sizes=(13,), activation='tanh', solver='lbfgs', max_iter=50000)
mlp.fit(Xtrain,Ytrain)

pred = mlp.predict(X_pred)
pred = pd.DataFrame(pred)
pred = pred.values
#print(pred)
np.savetxt("Y2_gr_U.csv", pred,  fmt='%1.1e')