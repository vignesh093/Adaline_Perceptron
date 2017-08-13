import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Perceptron(object):
    def __init__(self,n_lea_rate,n_iter):
        self.n_lea_rate=n_lea_rate
        self.n_iter=n_iter

    def get_activation(self,X):
        return (np.dot(self.w[1:],X)+self.w[0])

    def predict(self, X):
        activation = self.get_activation(X)
        return np.where(activation >= 0.0, 1, -1)

    def fit(self,X,Y):
        self.w=np.zeros(X.shape[1]+1)
        self.err=[]
        for i in range(self.n_iter):
            errors=0
            for inp,clas_val in zip(X,Y):
                predicted_val = self.predict(inp)
                error=clas_val-predicted_val
                self.w[1:]+=self.n_lea_rate*error*inp
                self.w[0]+=self.n_lea_rate * error
                errors+=int(error!=0.0)
            self.err.append(errors)
        print(self.err)














df = pd.read_csv('D:\\Users\\vignesh.i\\Desktop\\iris.csv', header=None)
X=df.iloc[0:100,[0,2]].values
Y=df.iloc[0:100,4].values
Y=np.where(Y == 'Iris-setosa',-1,1)
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

ppn=Perceptron(0.1,10)
ppn.fit(X,Y)
plt.plot(range(1,len(ppn.err)+1),ppn.err,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
