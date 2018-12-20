import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('X1_t1.csv') 
colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
#print(dataset.shape)
#print(dataset.head())


X = dataset.drop('ccs',axis=1).values
y = dataset['ccs'].values


scaler = MinMaxScaler(feature_range=(0, 1))

X_n = scaler.fit_transform(X)


#PLot the target VS each features (normalized)
plt.figure(figsize=(14,6))
for i in range(8):
    plt.subplot(3,3,i+1)
    plt.scatter(X_n[:,i],y,s=10)
    plt.title(colom[i])   
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.40,
                wspace=0.1)
plt.savefig('scatterFeatures.png')

#estimation of mutual information
def distribution(X):
    dim = X.ndim
    size = X.shape
    count = {}
    if dim == 1:
        for i in range(size[0]):
            entry = X[i]
            if entry not in count:
                count[entry] = 0
            count[entry] += 1
    else:           
        for i in range(size[0]):
            entry = tuple(X[i,:])
            if entry not in count:
                count[entry] = 0
            count[entry] += 1
    return {k:v/size[0] for (k,v) in count.items()}

def entropy(distr):
    p = np.array([v for v in distr.values()])
    return - (p * np.log2(p)).sum()

def concat(X1,X2):
    return np.concatenate(([X1],[X2]),axis=0).T

# and display + cluster on a heatmap
nf = 8
corr = np.zeros((nf,nf))

for i in range(nf):
    for j in range(nf):
        d12 = distribution(concat(X[:,i],X[:,j]))
        d1  = distribution(X[:,i])
        d2  = distribution(X[:,j])
        E   = entropy(d1) + entropy(d2)
        corr[i,j] = (E - entropy(d12)) / entropy(d12)
        
sns.clustermap(corr, annot=True, xticklabels = colom, yticklabels = colom);
plt.savefig('mutualInformation.png')

corry = np.zeros(nf)
for i in range(nf):
    d12 = distribution(concat(X[:,i],y))
    d1  = distribution(X[:,i])
    d2  = distribution(y)
    E   = entropy(d1) + entropy(d2)
    corry[i] = (E - entropy(d12)) / entropy(d12)

#width = 1/1.5
#plt.subplots(1)
#plt.figure(figsize=(10,5))
##plt.grid(axis = "y")
#plt.bar(colom, corry, width, color="green")
#plt.ylabel("Mutual information")
#plt.savefig('mutualInformationTarget.png')

for i in range(8):
    print("MI of %s is: %f" % (colom[i], corry[i]))
    

        