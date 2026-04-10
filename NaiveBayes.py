import numpy as np 
from Metrices.Evaluation import Accuracy
from sklearn.model_selection import train_test_split
from sklearn import datasets



class NaiveBayes():

    def __init__(self):
        pass

    def fit(self,X:np.ndarray,y:np.ndarray) -> None:
        n_samples , n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes,n_features))
        self.var = np.zeros((n_classes,n_features))
        self.prior = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            #! == P(Y) ==
            self.prior[idx] = X_c.shape[0] / n_samples 


    def predict(self, X:np.ndarray) -> np.ndarray:
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x:np.ndarray) -> np.ndarray:
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx:int, x:np.ndarray) -> float:
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator



if __name__ == "__main__":


    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print(f"Naive Bayes classification accuracy: {Accuracy(Predicted=predictions,Actual=y_test) * 100:0.2f} %" )
