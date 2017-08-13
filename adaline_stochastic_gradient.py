import numpy as np
import pandas as pd

class AdalineST(object):
    def __init__(self,l_rate,n_iter,shuffle):
        self.l_rate=l_rate
        self.n_iter=n_iter
        self.shuffle=shuffle

    def fit(self,X,Y):
        self.initialize_weights(X.shape[1])
        cost_whole=[]
        for i in range(self.n_iter):
            if self.shuffle:
                X,Y=self.shuffle_data(X,Y)
            cost_periter=[]
            for xdata,ydata in zip(X,Y):
                result=self.activation(xdata,ydata)
                error=ydata-result
                self.w[1:]+=np.dot(xdata,error)*self.l_rate
                self.w[0]+=self.l_rate*error
                cost=(error**2)/2
                cost_periter.append(cost)
            avg_cost=sum(cost_periter)/len(cost_periter)
            cost_whole.append(avg_cost)
        return cost_whole


    def shuffle_data(self,X,Y):
        per=np.random.permutation(len(Y))
        return X[per],Y[per]
    def initialize_weights(self,shape_X):
        self.w=np.zeros(shape_X+1)

    def activation(self,X,Y):
        return np.dot(X,self.w[1:])+self.w[0]


df = pd.read_csv('D:\\Users\\vignesh.i\\Desktop\\iris.csv', header=None)
X = df.iloc[0:100, [0, 2]].values
Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)
ad=AdalineST(0.01,50,True)
print(ad.fit(X,Y))
