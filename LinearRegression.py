import numpy as np 

class LinearRegression():
    def __init__(self,Learning_rate:float= 0.001, iterations:int = 1000):
        self.Learning_rate = Learning_rate 
        self.iterations = iterations
        self.weights = None
        self.bais = None

    

    def fit(self,X:np.ndarray,Y:np.ndarray) -> None:
        n_samples , n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bais = np.random.rand(1)

        for _ in range(self.iterations):
            y_predicted = np.dot(X,self.weights) + self.bais

            dw = (1 / n_samples) * (X.T @ (y_predicted - Y))
            db = (1 / n_samples) * np.sum(y_predicted - Y)

            self.weights = self.weights - (self.Learning_rate * dw)
            self.bais = self.bais - (self.Learning_rate * db)


    def predict(self,x) -> np.ndarray:
        return np.dot(x,self.weights) + self.bais
        




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from Metrices.Evaluation import RMSE , MSE

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = LinearRegression(Learning_rate=0.01, iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    

    print(f"MSE Error = {MSE(predictions,y_test)}")
    print(f"RMSE Error = {RMSE(predictions,y_test)}")

    plt.figure(figsize=(20,12))
    plt.scatter(X_train,y_train,marker="o")
    plt.scatter(X_test,y_test,marker="o",alpha=0.5)
    plt.plot(X,regressor.predict(X),color="red",linewidth=2)
    plt.show()

