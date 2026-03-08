import numpy as np 

def Accuracy(Predicted:np.ndarray|list,Actual:np.ndarray|list) -> float:
        return np.sum(Predicted == Actual) / len(Actual)

def MSE(Predicted:np.ndarray|list,Actual:np.ndarray|list) -> float:
        return np.mean((Predicted - Actual)**2)

def RMSE(Predicted:np.ndarray|list,Actual:np.ndarray|list) -> float:
        return (np.mean((Predicted - Actual)**2)) ** 0.5