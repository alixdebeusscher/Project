from tools import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#Read test file
X1=pd.read_csv("X1_t1.csv")
#column name
col = list(X1)
#Split test file in learning set and test set
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values

X_n = normalize(X)
#clf = MLPRegressor(alpha=0.0001, hidden_layer_sizes = (50,), max_iter = 50000, 
#                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
#a = clf.fit(Xtrain, Ytrain)
#
#pred_y = clf.predict(Xtest) # predict network output given x_
#fig = plt.figure() 
#plt.plot(Xtest, Ytest, color = 'b') # plot original function
#plt.subplots(1)
#plt.plot(Xtest, pred_y, '-') # plot network output

mlp = MLPRegressor(max_iter=2000)

#parameter_space = {
#    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#    'activation': ['tanh', 'relu'],
#    'solver': ['sgd', 'adam'],
#    'alpha': [0.0001, 0.05],
#    'learning_rate': ['constant','adaptive'],
#}
#parameter_space = {
#    'hidden_layer_sizes': [(50,)],
#    'verbose': ['True'],
#    'activation': ['logistic'],
#    'alpha': [0.0001],
#    'learning_rate': ['adaptive'],
#}


parameter_space = {
    'solver': ['sgd', 'adam', 'lbfgs'],
    'activation' : ['identity', 'logistic', 'tanh', 'relu']
}


from sklearn.model_selection import GridSearchCV
t = time.time()
# do stuff
clf = GridSearchCV(mlp, parameter_space, cv=5, scoring = score_function_neg)
print('coucou')
f = time.time()
clf.fit(X_n, y)
el = time.time() - f
print(el)
print('coucou')

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
 
    

############# NUMBER NEURONS #############  
mlp = MLPRegressor(max_iter=2000,solver='lbfgs',activation='relu')

NumberNeurons1 = []
NumberNeurons2 = []
NumberNeurons3 = []
for i in range(10,61):
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

clfNbrN1 = GridSearchCV(mlp, parameter_space1, cv=5, scoring = score_function_neg)
clfNbrN2 = GridSearchCV(mlp, parameter_space2, cv=5, scoring = score_function_neg)
clfNbrN3 = GridSearchCV(mlp, parameter_space3, cv=5, scoring = score_function_neg)
f = time.time()
clfNbrN1.fit(X_n, y)
el = time.time() - f
print(el)
f = time.time()
clfNbrN2.fit(X_n, y)
el = time.time() - f
print(el)
f = time.time()
clfNbrN3.fit(X_n, y)
el = time.time() - f
print(el)

# Best paramete set
print('Best parameters found:\n', clfNbrN1.best_params_)
print('Best parameters found:\n', clfNbrN2.best_params_)
print('Best parameters found:\n', clfNbrN3.best_params_)

# All results
means1 = clfNbrN1.cv_results_['mean_test_score']
stds1 = clfNbrN1.cv_results_['std_test_score']
means2 = clfNbrN2.cv_results_['mean_test_score']
stds2 = clfNbrN2.cv_results_['std_test_score']
means3 = clfNbrN3.cv_results_['mean_test_score']
stds3 = clfNbrN3.cv_results_['std_test_score']
train_sizes = means1.shape

plt.figure()
plt.title('Test')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(range(1,52), means1 - stds1,
                 means1 + stds1, alpha=0.1,
                 color="r")
plt.fill_between(range(1,52), means2 - stds2,
                 means2 + stds2, alpha=0.1, color="g")
plt.fill_between(range(1,52), means3 - stds3,
                 means1 + stds3, alpha=0.1,
                 color="b")
plt.plot(range(1,52), means1, color="r",
         label="One layer")
plt.plot(range(1,52), means2, color="g",
         label="Three layers")
plt.plot(range(1,52), means3, color="b",
         label="Five layers")
plt.legend(loc="best")

    
    
    
############# NUMBER LAYERS #############  
NumberLayers1 = []
NumberLayers2 = []
NumberLayers3 = []
for i in range(1,11):
    NumberLayers1.append((15,)*i)
    NumberLayers2.append((30,)*i)
    NumberLayers3.append((45,)*i)
    
parameter_space1 = {
    'hidden_layer_sizes': NumberLayers1,
}
parameter_space2 = {
    'hidden_layer_sizes': NumberLayers2,
}
parameter_space3 = {
    'hidden_layer_sizes': NumberLayers3,
}

clfNbrL1 = GridSearchCV(mlp, parameter_space1, cv=5, scoring = score_function_neg)
clfNbrL2 = GridSearchCV(mlp, parameter_space2, cv=5, scoring = score_function_neg)
clfNbrL3 = GridSearchCV(mlp, parameter_space3, cv=5, scoring = score_function_neg)
f = time.time()
clfNbrL1.fit(X_n, y)
el = time.time() - f
print(el)
f = time.time()
clfNbrL2.fit(X_n, y)
el = time.time() - f
print(el)
f = time.time()
clfNbrL3.fit(X_n, y)
el = time.time() - f
print(el)

# Best paramete set
print('Best parameters found:\n', clfNbrL1.best_params_)
print('Best parameters found:\n', clfNbrL2.best_params_)
print('Best parameters found:\n', clfNbrL3.best_params_)