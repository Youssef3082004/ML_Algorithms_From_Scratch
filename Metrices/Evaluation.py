import numpy as np 

def Accuracy(Predicted:np.ndarray|list,Actual:np.ndarray|list) -> float:
        return np.sum(Predicted == Actual) / len(Actual)

def MSE(Predicted:np.ndarray|list,Actual:np.ndarray|list) -> float:
        return np.mean((Predicted - Actual)**2)

def RMSE(Predicted:np.ndarray|list,Actual:np.ndarray|list) -> float:
        return (np.mean((Predicted - Actual)**2)) ** 0.5

def MAE(Predicted:np.ndarray|list,Actual:np.ndarray|list) ->float:
        return (np.mean(np.abs(Predicted - Actual)))

def R2_Score(Predicted:np.ndarray|list,Actual:np.ndarray|list) ->float:
        y_mean = np.mean(Actual)
        formula = np.sum((Actual - Predicted) ** 2) / np.sum((Actual - y_mean) ** 2)
        return 1 - formula