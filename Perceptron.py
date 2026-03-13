import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Metrices.Evaluation import Accuracy
from sklearn.datasets import load_breast_cancer

class Perceptron():

    def __init__(self,Learning_rate:float = 0.001, iterations:int = 1000):
        self.Learning_rate = Learning_rate 
        self.iterations = iterations
        self.weights = None
        self.loss = [0] * self.iterations
    

    def fit(self,X:np.ndarray,y:np.ndarray):
        n_samples , n_features = X.shape
        self.weights = np.random.random(n_features + 1)
        self.X_biased = np.column_stack([X,np.ones(n_samples)])

        for epoch in range(self.iterations):
            error_count = 0
            for i in range(n_samples):
                sample = self.X_biased[i]

                linear_model = sample @ self.weights
                error = self._step(linear_model) - y[i]
                self.weights -= self.Learning_rate * error * sample

                if error != 0: error_count+=1
            self.loss[epoch] = error_count

            if error_count == 0:
                print(f"Converged at epoch {epoch + 1}!")  
                break   

    def predict(self,X:np.ndarray):
        y_predtiction = np.column_stack([X,np.ones(X.shape[0])]) @ self.weights
        return self._step(y_predtiction)        


    def _step(self,z):
        return np.where(z < 0, 0 , 1 ) 
    
    def plot_loss(self):
        plt.plot(range(1, self.iterations + 1), self.loss, color='red', linewidth=1)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Number of Misclassifications', fontsize=12)
        plt.title('Perceptron: Training Errors', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
   
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    p = Perceptron(Learning_rate=0.01, iterations=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print(f"Perceptron classification accuracy: {Accuracy(Predicted=predictions,Actual=y_test) * 100:0.2f} %")

    p.plot_loss()