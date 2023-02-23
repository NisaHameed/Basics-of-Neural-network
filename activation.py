import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):

        self.A = 1/(1 + np.exp(-Z))

        return self.A

    def backward(self):

        dAdZ = self.A * (1  - self.A)

        return dAdZ

class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward (self,Z):
        self.A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        return self.A
    
    def backward(self):
        dAdZ = 1 - self.A ** 2
        return dAdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self,Z):
        self.A = np.maximum(0, Z)
        return self.A
    
    def backward(self):
        
        dAdZ = self.A.copy()
        
        dAdZ[dAdZ==0] = 0
        dAdZ[dAdZ>0] = 1
        
        return dAdZ