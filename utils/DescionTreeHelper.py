import numpy as np 

def _Entropy(y:np.ndarray) -> np.float64:
        classes_count = np.bincount(y)
        if len(classes_count) == 1:
            return 0
        ps = classes_count / len(y)
        return - np.sum([(p * _log(p,len(classes_count))) for p in ps if p > 0])

def _log(x, base=np.e):
        return np.log(x) / np.log(base)
         

def Information_Gain(x_column:np.ndarray,y:np.ndarray) -> np.float64:
        parent_entropy = _Entropy(y)
        weighted_child_entropy = 0

        for value in np.unique(x_column):

            subset = y[x_column == value]
            entropy = _Entropy(subset)
            probability = len(subset) / len(x_column)
            weighted_child_entropy += (probability * entropy)  
        
        return (parent_entropy - weighted_child_entropy)

def Best_split(X: np.ndarray, Y: np.ndarray, Features_indices: list) -> tuple[float, int]:

    best_ig = -1
    best_feat_split = None

    for index in Features_indices:
        ig = Information_Gain(X[:, index], Y)

        if ig > best_ig:
            best_ig = ig
            best_feat_split = index

    return best_ig, best_feat_split