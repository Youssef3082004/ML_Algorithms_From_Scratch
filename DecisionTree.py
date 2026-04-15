import numpy as np 
from utils.DescionTreeHelper import Best_split
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Metrices.Evaluation import Accuracy

 


class Node():
    def __init__(self, data= None, feature_idx= None, feature_val= None, prediction_probs=None, information_gain=None) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.children = {}



class DecisionTree():

    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def fit(self,X:np.ndarray,Y:np.ndarray) -> None:
        self.root = self._Build_tree(X=X,Y=Y,Features_indices=[x for x in range(X.shape[-1])],depth=self.max_depth)

    def predict(self,X:np.ndarray) -> np.ndarray:
        return np.array([self._Traverse_Tree(x=x,node=self.root) for x in X  ])


    
    def _Build_tree(self,X:np.ndarray,Y:np.ndarray,Features_indices:list,depth:int = 0) -> Node:
        Y = Y.flatten()
        classes = np.unique(Y)

        if len(classes) == 1 or len(Features_indices) == 0:
            majority_class = Counter(Y).most_common(1)[0][0]
            return Node(prediction_probs=majority_class)

        best_ig, feature_idx = Best_split(X=X, Y=Y, Features_indices=Features_indices)

        if best_ig == 0 or feature_idx is None:
            majority_class = Counter(Y).most_common(1)[0][0]
            return Node(prediction_probs=majority_class)

        node = Node(feature_idx=feature_idx, information_gain=best_ig)
        
        remaining_features = [f for f in Features_indices if f != feature_idx]

        for value in np.unique(X[:, feature_idx]):
            y_subset = Y[X[:, feature_idx] == value]
            x_subset = X[X[:, feature_idx] == value]
            child_node = self._Build_tree(x_subset, y_subset, remaining_features, depth + 1)
            child_node.feature_val = value
            node.children[value] = child_node

        return node
    


    def _Traverse_Tree(self,x:np.ndarray,node:Node):
        if not node.children:
            return node.prediction_probs

        feature_idx = node.feature_idx
        feature_value = x[feature_idx]
        
        if feature_value in node.children:
            return self._Traverse_Tree(x,node.children[feature_value])
        else:
            return None



if __name__ == "__main__":

    data = datasets.load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = Accuracy(y_test, predictions)

    print(f"DecisionTree Classifier Accuracy: {Accuracy(Predicted=predictions,Actual=y_test) * 100:0.2f} %")
