import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]
        se = np.multiply(np.subtract(A,Y),np.subtract(A,Y))
        self.identity_N = np.ones((self.N,1))
        self.identity_C = np.ones((self.C,1))
        sse = self.identity_N.T @ se @ self.identity_C
        mse = sse/(2*self.N*self.C)

        return mse

    def backward(self):

        dLdA = np.subtract(self.A,self.Y) / (self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]
        C = A.shape[1]

        Ones_C = np.ones((C,1))
        Ones_N = np.ones((N,1))

        e = np.exp(A)
        sum=e.sum(axis=1)
        sum=sum.reshape(N,1)
        self.softmax=e/sum
        log_softmax = np.log(self.softmax)
        crossentropy = np.multiply(-Y,log_softmax) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = np.subtract(self.softmax , self.Y)

        return dLdA
