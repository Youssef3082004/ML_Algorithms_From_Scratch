import numpy as np 
from Metrices.Distance import Distances 
from collections import Counter


class KNN():

    def __init__(self,k:int=3):
        self.k = k
    

    def fit(self,x,y) -> None:
        self.X = x 
        self.Y = y
    

    def predict(self,x_test) -> np.ndarray:
        y_pred = [self._predict(x) for x in x_test]
        return np.array(y_pred)


    def _predict(self,x) -> int:
        #! Get Distances between every point from x_test to every point in x_train 
        distances = [Distances.Euclidean(x,x_train) for x_train in self.X]

        #! Sorting Distances for each array and get most K Classes
        sorted_distances_indices = np.argsort(distances)[:self.k]

        #! Get classes Labels 
        classes = [self.Y[class_index] for class_index in sorted_distances_indices]

        return Counter(classes).most_common(1)[0][0]