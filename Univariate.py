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

dataset = pd.read_csv('X1_t1.csv') 
colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
#print(dataset.shape)
#print(dataset.head())

def normalize_data(X):
    # get the shape of this data (#features)
    nd, nf = X.shape

    for i in range(nf):
        X_n[:,i] = (X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())
    return X_n


X = dataset.drop('ccs',axis=1).values
y = dataset['ccs'].values

X_n = normalize_data(X)
#y_n = normalize_data(y)

#PLot the target VS each features (normalized)
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(4,2,2*i+1)
    plt.scatter(X_n[:,2*i],y)
    plt.title(colom[2*i])
    plt.subplot(4,2,2*i+2)
    plt.scatter(X_n[:,2*i+1],y)
    plt.title(colom[2*i+1])
    
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.85,
                wspace=0.2)


# estimation of mutual information
def distribution(X, *columns):
    nd, _ = X.shape
    count = {}
    for i in range(nd):
        entry = tuple(X[i, columns])
        if entry not in count:
            count[entry] = 0
        count[entry] += 1
    return {k:v/nd for (k,v) in count.items()}

#def distribution(X):
#    dim = X.ndim
#    size = X.shape
#    count = {}
#    if dim == 1:
#        for i in range(size[0]):
#            entry = X[i]
#            if entry not in count:
#                count[entry] = 0
#                count[entry] += 1
#    else:           
#        for i in range(size[0]):
#            entry = tuple(X[i,:])
#            if entry not in count:
#                count[entry] = 0
#                count[entry] += 1
#    return {k:v/size[0] for (k,v) in count.items()}

def entropy(distr):
    p = np.array([v for v in distr.values()])
    return - (p * np.log2(p)).sum()

def normalized_mi(X, c1, c2):
    d12 = distribution(X, c1, c2)
    d1  = distribution(X, c1)
    d2  = distribution(X, c2)
    E   = entropy(d1) + entropy(d2)
    return (E - entropy(d12)) / entropy(d12)

def concat(X1,X2):
    return np.concatenate(([X1],[X2]),axis=0).T

# and display + cluster on a heatmap
nf = 8
corr = np.zeros((nf,nf))

for i in range(nf):
    for j in range(nf):
#        d12 = distribution(concat(X[:,i],X[:,j]))
#        d1  = distribution(X[:,i])
#        d2  = distribution(X[:,j])
#        E   = entropy(d1) + entropy(d2)
#        corr[i,j] = (E - entropy(d12)) / entropy(d12)
        corr[i,j] = normalized_mi(X, i, j)
        
labels = colom
labels = ['#%d' % i for i in range(1, 9)]
sns.clustermap(corr, annot=True, xticklabels = labels, yticklabels = labels);
#plt.savefig('mutualinformation.png', dpi=300)

corry = np.zeros((nf,1))
for i in range(nf):
    d12 = distribution(concat(X[:,i],y))
    d1  = distribution(X[:,i])
    d2  = distribution(y)
    E   = entropy(d1) + entropy(d2)
    corry[i] = (E - entropy(d12)) / entropy(d12)


plt.scatter(range(1,9),corry)
        