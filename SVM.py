import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Metrices.Evaluation import Accuracy


class SupportVectorMachine():

    def __init__(self,Learning_rate:float= 0.001,lambda_param=0.01,iterations:int = 1000):
        self.Learning_rate = Learning_rate 
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.loss = [0] * self.iterations

    def fit(self,X:np.ndarray,y:np.ndarray) -> None:
        n_samples , n_features = X.shape
        self.weights = np.random.rand(n_features + 1)
        self.x_bias = np.column_stack([X,-np.ones(n_samples)])


        for epoch in range(self.iterations):
            for i in range(n_samples):
                sample = self.x_bias[i]
                condition = y[i] * (sample @ self.weights) >= 1
                gradient =  2 * self.lambda_param * self.weights if condition else (2 * self.lambda_param * self.weights) - (sample * y[i])
                self.weights -= self.Learning_rate * gradient

            hinge_loss = np.maximum(0, 1 - (y * (self.x_bias @ self.weights)))
            avg_hinge_loss = np.mean(hinge_loss)
            
            l2_reg_loss = self.lambda_param * np.sum(self.weights**2)

            total_loss = (avg_hinge_loss / n_samples) + l2_reg_loss
            self.loss[epoch] = total_loss
            
            if total_loss <=  5e-3:
                print(f"Stopped in epoch {epoch} - Loss reached 0")
                break

    def predict(self,x:np.ndarray) -> np.ndarray:
        linear_model = np.column_stack([x,-np.ones(x.shape[0])]) @ self.weights 
        return np.sign(linear_model)
    
    def plot_loss(self) -> None:
        plt.plot(range(1, self.iterations + 1), self.loss, color='red', linewidth=1)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Number of Misclassifications', fontsize=12)
        plt.title('Perceptron: Training Errors', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    

X, y = datasets.load_breast_cancer(return_X_y=True)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SupportVectorMachine()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Support Vector Machine classification accuracy: {Accuracy(Predicted=predictions,Actual=y_test) * 100:0.2f} %")  

model.plot_loss()