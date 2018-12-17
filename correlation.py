import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
import seaborn as sns
#Read test file
X1=pd.read_csv("X1_t1.csv")

#Split test file in learning set and test set
X = X1.drop('ccs',axis=1).values
y = X1['ccs'].values
colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']
nf = 8
corr = np.zeros((nf,nf))

def entropy(distr):
    p = np.array([v for v in distr.values()])
    return - (p * np.log2(p)).sum()

def distribution(X, *columns):
    'compute distribution with explicit bucketing'
    nd, _ = X.shape
    count = {}
    for i in range(nd):
        entry = tuple(X[i, colom])
        if entry not in count:
            count[entry] = 0
        count[entry] += 1
    return {k:v/nd for (k,v) in count.items()}

def normalized_mi(X, c1, c2):
    d12 = distribution(X, c1, c2)
    d1  = distribution(X, c1)
    d2  = distribution(X, c2)
    E   = entropy(d1) + entropy(d2)
    return (E - entropy(d12)) / entropy(d12)

for i in range(nf):
    for j in range(nf):
        corr[i,j] = normalized_mi(X1, i, j)
        
labels = colom
labels = ['#%d' % i for i in range(1, 9)]
sns.clustermap(corr, annot=True, xticklabels = labels, yticklabels = labels);
mp.savefig('mutualinformation.png', dpi=300)

