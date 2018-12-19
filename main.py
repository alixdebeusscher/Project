import tools 
import methods

#read file
X1=pd.read_csv("X1_t1.csv")

#column name
col = list(X1)

#Split test file in learning set and test set
X = X1.drop(col[-1],axis=1).values
y = X1[col[-1]].values
X_scaled = tools.normalize(X)

