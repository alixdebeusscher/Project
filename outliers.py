import pandas as pd
import matplotlib.pyplot as plt
#Read test file
X1=pd.read_csv("X1_t1.csv")

#Split test file in learning set and test set
X = X1.drop('ccs',axis=1).values
y = X1['ccs'].values

colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']

X1_df=pd.DataFrame(X)
X1_df.columns = colom
#print(X1_df.head())

import seaborn as sns
#a=sns.boxplot(x=y)
sns.set(style="whitegrid")
tips = X1
#ax = sns.boxplot(x=tips["ccs"])
#ax = sns.swarmplot(x=tips["ccs"], color=".25")
print(len(y))
for i in range(len(y)):
 #  print(y[i])
    if y[i]>80.:
       print('row to deleted :', i)
       print(X[i])
       print(y[i])
      
ag = sns.boxplot(x=tips["ccs"])
ag = sns.swarmplot(x=tips["ccs"], color=".25")

#fig, ax =plt.subplots(3,3)
#sns.boxplot(y=tips['ccs'], ax=ax[0][0])
#sns.swarmplot(y=tips['ccs'], ax=ax[0][0], color=".25")
##sns.boxplot(y=tips['cement'], ax=ax[0][1])
##sns.swarmplot(y=tips['cement'], ax=ax[0][1], color=".25")
#sns.boxplot(y=tips['blast'], ax=ax[0][2])
##sns.swarmplot(y=tips['blast'], ax=ax[0][2], color=".25")
#sns.boxplot(y=tips['fly'], ax=ax[1][0])
##sns.swarmplot(y=tips['fly'], ax=ax[1][0], color=".25")
#sns.boxplot(y=tips['water'], ax=ax[1][1])
##sns.swarmplot(y=tips['water'], ax=ax[1][1], color=".25")
#sns.boxplot(y=tips['superplas'], ax=ax[1][2])
##sns.swarmplot(y=tips['superplas'], ax=ax[1][2], color=".25")
#sns.boxplot(y=tips['coarse_agg'], ax=ax[2][0])
##sns.swarmplot(y=tips['coarse_agg'], ax=ax[2][0], color=".25")
#sns.boxplot(y=tips['fine_agg'], ax=ax[2][1])
##sns.swarmplot(y=tips['fine_agg'], ax=ax[2][1], color=".25")
#sns.boxplot(y=tips['age'], ax=ax[2][2])
##sns.swarmplot(y=tips['age'], ax=ax[2][2], color=".25")
fig.show()