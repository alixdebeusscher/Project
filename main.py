from methods import *

#read file
X1=pd.read_csv("X1_t1.csv")

#Column name + target
col = list(X1)

#Load the data
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values

#Remove the target name
col = col[:-1]

#Normalized input
X_n = normalize(X)

#Do the plot for the data analysis + MI
methods.data_analysis(X,X_n,y,col)

#Plot for the linear model + coef
linear(X_n,y,col)

<<<<<<< HEAD
kNN(X,y,col)
=======
methods.kNN(X,y,col)

methods.mlp_solver_activation(X_n,y,col)

solver = ['lbfgs']
activation = ['tanh','relu']
n_neur_min = 10
n_neur_max = 20
n_layer_min = 1
n_layer_max = 3


result = mlp_final(X_n, y, solver, activation, n_neur_min, n_neur_max,n_layer_min,n_layer_max)
print(result)
>>>>>>> c7a4c5e43782627d3f3fa96d5800ba3b40be2035
