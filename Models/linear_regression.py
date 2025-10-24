import numpy as np


class Linear_Regression:
    def fit(self,X,Y,lr = 0.05,iter=1000):
        params = np.zeros((X.shape[1]+1,1))
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X_std[self.X_std == 0] = 1
        X = (X-self.X_mean)/self.X_std
        X = np.c_[np.ones((X.shape[0], 1)), X]
        m = X.shape[0]
        cost_history = []
        for i in range(iter):
            pred = X@params
            error = pred-Y
            grad = (X.T@(error))/m
            cost = np.sum(error**2)/(2*m)
            cost_history.append(cost)
            params = params - lr*(grad)
        self.params = params
        self.costs = cost_history
        return self

    def predict(self,X):
        X = (X-self.X_mean)/self.X_std
        X = np.c_[np.ones((X.shape[0], 1)), X]
        pred = X@self.params
        return pred

if __name__=="__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Linear_Regression().fit(X_train,y_train)