import numpy as np 


class LogisticRegression():

    def __init__(self,Learning_rate:float= 0.001, iterations:int = 1000):
        self.Learning_rate = Learning_rate 
        self.iterations = iterations
        self.weights = None
        self.bais = None

    def fit(self,X:np.ndarray,y:np.ndarray):
        n_samples , n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bais = np.random.rand(1)

        for _ in range(self.iterations):
            y_predicted = (X @ self.weights) + self.bais
            
            dw = (1 / n_samples) * ( X.T @ (self._sigmoid(y_predicted) - y))
            db = (1 / n_samples) * np.sum(self._sigmoid(y_predicted) - y)

            self.weights = self.weights - (self.Learning_rate * dw)
            self.bais = self.bais - (self.Learning_rate * db)


    def predict(self,x) -> np.ndarray:
        linear_model = np.dot(x,self.weights) + self.bais
        y_predicted = self._sigmoid(linear_model)
        return [0 if y < 0.5 else 1 for y in y_predicted]

        
    def _sigmoid(self,z:np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from Metrices.Evaluation import Accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,)

    regressor = LogisticRegression(Learning_rate=0.001, iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print(f"LR classification accuracy: {Accuracy(Predicted=predictions,Actual=y_test) * 100:0.2f} %" )