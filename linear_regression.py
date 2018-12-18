import pandas as pd
import matplotlib.pyplot as plt 
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from math import sqrt


dataset = pd.read_csv('X1_t1.csv') 
colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
dataset.shape
#print(dataset.head())

X = dataset.drop('ccs',axis=1).values
y = dataset['ccs'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
x_test = pd.DataFrame(x_test_scaled)

regressor = LinearRegression()
  
regressor.fit(X_train, y_train)  
print('EIC', regressor.intercept_)
print('NOC', len(regressor.coef_))
pred = regressor.predict(X_test)  
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
print('error is', error)

def score_function(model, X, Y):
    'norm-2 criterion for optimization of models'
    return numpy.sqrt(numpy.mean(((model.predict(X) - Y)) **2, axis=0)).sum()

# K-fold cross-validation score of a model
def cross_validate(model, data, target, cv = 10, score=score_function, doprint=False):
    'apply k-fold cross validation on model (returns mean and std of tests'
    'if doprint is True, this also does a nice display of the result'
    scores = cross_val_score(model, data, target, cv=cv, scoring=score)
    if doprint:
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (scores.mean(), scores.std())


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
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

fig = plot_learning_curve(regressor, 'Linear regression cooling load learning curve', 
                          X,y, cv=10)
fig.tight_layout()

    