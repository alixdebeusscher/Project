import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
from math import sqrt


dataset = pd.read_csv('X1_t1.csv') 
colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
dataset.shape
#print(dataset.head())

X = dataset.drop('ccs',axis=1).values
y = dataset['ccs'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)

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

k = 0
tab = [0, 0, 0, 0, 0, 0, 0, 0]
for each in colom:
    for i in range(10000):
        tab[k]+=regressor.coef_[k]/10000
    print("ECoef of %s is: %f", each, tab[k])        
    k+=1

y = tab
N = len(y)
x = range(N)
width = 1/1.5
plt.bar(x, y, width, color="blue")


fig = plt.gcf()

    