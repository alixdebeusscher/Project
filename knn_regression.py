import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as mp
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def score_function(model, X, Y):
    'norm-2 criterion for optimization of models'
    return np.sqrt(np.mean(((model.predict(X) - Y)) **2, axis=0)).sum()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20), scoring=score_function):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt = mp.figure()
    mp.title(title)
    if ylim is not None:
        mp.ylim(*ylim)
    mp.xlabel("Training examples")
    mp.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    mp.grid()

    mp.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    mp.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    mp.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    mp.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    mp.legend(loc="best")
    return plt

#Read test file
X1=pd.read_csv("X1_t1.csv")

#Split test file in learning set and test set
X = X1.drop('ccs',axis=1).values
y = X1['ccs'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, shuffle=True)

#D'autres moyen de normaliser ?????????? je comprends pas trop cette section
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
x_test = pd.DataFrame(x_test_scaled)

    
rmse_val = [] #to store rmse values for different k
for K in range(22):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    

plot_learning_curve(model, "title", X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20), scoring=score_function)
#x axis pas ok
curve = pd.DataFrame(rmse_val) #elbow curve 
plt = curve.plot()

k = min(rmse_val)
print(k)
#grpah commence à 0 mais c'est faux
X2=pd.read_csv("X2.csv")
pred = model.predict(X2) 
#print(pred)
#save pred to csv

