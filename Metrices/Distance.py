import numpy as np 



class Distances():

    @staticmethod
    def Euclidean(x1,x2):

        return np.sqrt(np.sum((x1 - x2)**2))