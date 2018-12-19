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
from sklearn.feature_selection import RFECV

#Read test file
X1=pd.read_csv("X1_t1.csv")
#column name
col = list(X1)
col.pop(-1)
#Split test file in learning set and test set
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values

X_n = normalize(X)

mlp = MLPRegressor(max_iter=2000)


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
 
    

#%% NUMBER NEURONS  
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

#print('Best parameters found:\n', clfNbrL1.best_params_)
#means = -clfNbrL1.cv_results_['mean_test_score']
#stds = -clfNbrL1.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clfNbrL1.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


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

#%% PLot
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
    
    
#%% NUMBER LAYERS
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

# All results
means1 = -clfNbrL1.cv_results_['mean_test_score']
stds1 = -clfNbrL1.cv_results_['std_test_score']
means2 = -clfNbrL2.cv_results_['mean_test_score']
stds2 = -clfNbrL2.cv_results_['std_test_score']
means3 = -clfNbrL3.cv_results_['mean_test_score']
stds3 = -clfNbrL3.cv_results_['std_test_score']

#%% PLot
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




#%% Test
mlpFeatures = MLPRegressor(max_iter=2000)
parameter_space = {
        'hidden_layer_sizes': [(12,),(13,),(14,)],
        'solver': ['lbfgs'],
        'activation' : ['tanh'],
    }

 
 
clfFeatures = GridSearchCV(mlpFeatures, parameter_space, cv=5, scoring = score_function_neg)
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
    f = time.time()
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
    print('Time : ', time.time()-f)
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

#%%
model = MLPRegressor(max_iter=2000,activation = 'tanh', 
                     hidden_layer_sizes = (13,),solver = 'lbfgs')
plot_learning_curve(model, '', X_n, y, cv=5)
plt.savefig('FinalMLP.png')
#%% 
score = cross_val_score(model,X_n,y, cv=5, scoring=score_function)
print(score.mean())
print(score.std())
