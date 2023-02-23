import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features,in_features))
        self.b = np.zeros((out_features,1))
        self.out_features = out_features
        
        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        
        self.N = A.shape[0] #  store the batch size of input
        Z=np.zeros((self.N,self.out_features))
   
        # Think how will self.Ones helps in the calculations and uncomment below
        # self.Ones = np.ones(3,1)
        for i in range (self.N):
            temp=(self.W @ (self.A[i,:]))
            temp=(temp.reshape(self.out_features,1))
            
            Z[i,:] =(temp+self.b).T
        print(Z)
        return Z

    def backward(self, dLdZ):

        dZdA = self.W
        
        dZdA = dZdA.T #transpose of the derivative
        dZdW = self.A
        N = self.A.shape[0]
        dZdb = np.ones((N,1))

        dLdA = dLdZ @ dZdA.T
        dLdW = dLdZ.T @ dZdW
        dLdb = dLdZ.T @ dZdb
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
