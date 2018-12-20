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

kNN(X,y,col)
