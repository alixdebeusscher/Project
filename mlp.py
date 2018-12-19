import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Read test file
X1=pd.read_csv("X1_t1.csv")
#column name
col = list(X1)
#Split test file in learning set and test set
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.8, random_state=0)

#clf = MLPRegressor(alpha=0.0001, hidden_layer_sizes = (50,), max_iter = 50000, 
#                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
#a = clf.fit(Xtrain, Ytrain)
#
#pred_y = clf.predict(Xtest) # predict network output given x_
#fig = plt.figure() 
#plt.plot(Xtest, Ytest, color = 'b') # plot original function
#plt.subplots(1)
#plt.plot(Xtest, pred_y, '-') # plot network output

mlp = MLPRegressor(max_iter=50000)

#parameter_space = {
#    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#    'activation': ['tanh', 'relu'],
#    'solver': ['sgd', 'adam'],
#    'alpha': [0.0001, 0.05],
#    'learning_rate': ['constant','adaptive'],
#}
parameter_space = {
    'hidden_layer_sizes': [(50,)],
    'verbose': ['True'],
    'activation': ['logistic'],
    'alpha': [0.0001],
    'learning_rate': ['adaptive'],
}

from sklearn.model_selection import GridSearchCV
t = time.time()
# do stuff
print('hellow')
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
elapsed = time.time() - t
print(elapsed)
f = time.time()
clf.fit(Xtrain, Ytrain)
el = time.time() - f
print(f)
print('coucou')

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
#y_true, y_pred = DEAP_y_test , clf.predict(DEAP_x_test)
#
#from sklearn.metrics import classification_report
#print('Results on the test set:')
#print(classification_report(y_true, y_pred))