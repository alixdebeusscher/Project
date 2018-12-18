import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#Read test file
X1=pd.read_csv("X1_t1.csv")

#Split test file in learning set and test set
X = X1.drop('ccs',axis=1).values
y = X1['ccs'].values

colom = ['cement','blast','fly','water','superplas','coarse_agg','fine_agg','age']

X1_df=pd.DataFrame(X)
X1_df.columns = colom
#print(X1_df.head())
scaler = MinMaxScaler()
#
x_train_scaled = scaler.fit_transform(X)
Xf = pd.DataFrame(x_train_scaled)
Xf.columns=colom

import seaborn as sns
#a=sns.boxplot(x=y)
sns.set(style="whitegrid")
tips = Xf
#ax = sns.boxplot(x=tips["ccs"])
#ax = sns.swarmplot(x=tips["ccs"], color=".25")
print(len(y))
for i in range(len(y)):
 #  print(y[i])
    if y[i]>80.:
       print('row to deleted :', i, '\n')
       print(tips.iloc[i])
       print(y[i])
      
#ag = sns.boxplot(x=y)
#ag = sns.swarmplot(x=y, color=".25")

#fig, ax =plt.subplots(3,3)
#sns.boxplot(y=y, ax=ax[0][0])
##sns.swarmplot(y=y, ax=ax[0][0], color=".25")
#sns.boxplot(y=tips['cement'], ax=ax[0][1])
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
#fig.show()

#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# rectangular box plot
#bplot1 = sns.boxplot(x=y)  # fill with color
#bplot1 = sns.swarmplot(x=y, color=".25")
#bplot1.set_xlabel('Boxplot of "ccs", target')
#bplot1.set_ylabel('ylabel')

# notch shape box plot
bplot2 = sns.boxplot(y=Xf)
#                         patch_artist=True)   # fill with color


# fill with colors
#colors = ['pink', 'lightblue', 'lightgreen', 'pink', 'lightblue', 'lightgreen', 'pink', 'lightblue']
#for patch, color in zip(bplot2, colors):
#       patch.set_facecolor(color)

# adding horizontal grid lines
#for ax in axes:
#    ax.yaxis.grid(True)
#    ax.set_xticks([y+1 for y in range(len(y))], )
#    ax.set_xlabel('xlabel')
#    ax.set_ylabel('ylabel')

# add x-tick labels
#plt.setp(axes, xticks=[y+1 for y in range(len(Xf))],
        # xticklabels=['x1', 'x2', 'x3', 'x4', 'b', 'n', 'n', 'b'])

#plt.show()