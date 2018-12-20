from methods import *

#%%Prepare file
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

#%%Plot for the data analysis + MI
methods.data_analysis(X,X_n,y,col)

#%%Linear Regression
linear(X_n,y,col)

#%%KNN
kNN(X,y,col)

#%%MLP 
mlp_solver_activation(X_n,y,col)

solver = ['lbfgs']
activation = ['tanh','relu']
n_neur_min = 10
n_neur_max = 20
n_layer_min = 1
n_layer_max = 3
methods.mlp_final(X_n, y, solver, activation, n_neur_min, n_neur_max,n_layer_min,n_layer_max)

<<<<<<< HEAD
solver = ['lbfgs']
activation = ['tanh','relu']
n_neur_min = 12
n_neur_max = 14
n_layer_min = 1
n_layer_max = 1
methods.mlp_select_features(X_n,y,col,solver, activation, n_neur_min, n_neur_max, n_layer_min, n_layer_max)

#%% Final prediction
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

mlp = MLPRegressor(hidden_layer_sizes=(13,), activation='tanh', solver='lbfgs', max_iter=50000)
mlp.fit(Xtrain,Ytrain)

pred = mlp.predict(X_pred)
pred = pd.DataFrame(pred)
pred = pred.values
#print(pred)
np.savetxt("Y2_gr_U.csv", pred,  fmt='%1.1e')
=======
mlp_final(X_n, y, solver, activation, n_neur_min, n_neur_max,n_layer_min,n_layer_max)

#%%Save prediction to Y2.csv with best model

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
mlp = MLPRegressor(hidden_layer_sizes=(13,), activation='tanh', solver='lbfgs', max_iter=50000)
mlp.fit(Xtrain,Ytrain)
pred = mlp.predict(X_pred)
pred = pd.DataFrame(pred)
pred = pred.values
np.savetxt("Y2.csv", pred,  fmt='%1.1e')
>>>>>>> 295a8f9d84511a6e6b2c5642b7e9238c41651dd0
