import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from Metrices.Evaluation import Accuracy , MSE
import matplotlib.pyplot as plt

class Adaline():

    def __init__(self,Learning_rate:float = 0.001, iterations:int = 1000):
        self.Learning_rate = Learning_rate 
        self.iterations = iterations
        self.weights = None
        self.loss = []    

    def fit(self,X:np.ndarray,y:np.ndarray) -> None:
        n_samples , n_features = X.shape
        self.weights = np.random.random(n_features+1)
        self.X = np.column_stack([X,np.ones(n_samples)])

        for _ in range(self.iterations):
            
            y_predicted = self.X @ self.weights 
            dw = (1 / n_samples) * (self.X.T @ (y_predicted - y))
            self.weights = self.weights - self.Learning_rate * dw 
            self.loss.append(MSE(Predicted=y_predicted,Actual=y))

    def predict(self,X:np.ndarray) -> np.ndarray:
        z = np.column_stack([X,np.ones(X.shape[0])]) @ self.weights
        return np.where(z >= 0.0, 1, -1)


    def plot_loss(self):
        plt.plot(range(1, self.iterations + 1), self.loss, 'red', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.title('Adaline: Training Cost', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    bc = datasets.load_breast_cancer()
    X, y = bc.data, np.where(bc.target == 0,-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = Adaline(Learning_rate=0.001, iterations=10000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accuracy = Accuracy(y_test, predictions)
    print(f"Adaline classification accuracy: {accuracy * 100:0.2f}%")

    # regressor.plot_loss()

