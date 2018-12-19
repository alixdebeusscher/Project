import tools

def linear(X,y):
    regressor = LinearRegression()
      
    regressor.fit(X, y)  
    print(regressor.coef_)
    
    width = 1/1.5
    plt.figure(figsize=(10,3))
    plt.bar(col[:-1], regressor.coef_, width, color="green")
    plt.savefig('coef.png')
    
    plot_learning_curve(LinearRegression(), '', X, y, cv=5)
    plt.savefig('lr.png')
