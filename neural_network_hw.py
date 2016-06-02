import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

import numpy as np



def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad


class Neural_Network(object):
    def __init__(self):
        # define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 10
        self.lambduh = .0001

        # weights (Parameters)
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propagate through the network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat


    def sigmoid(self, z):
        # apply sigmoid activation to scalar vector or array
        return z#1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # apply sigmoid activation to scalar vector or array
        return 1#z*(1-z)
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = (.5)* (np.sum((y-self.yHat)**2))/X.shape[0] + (self.lambduh/2)*(np.sum(self.W2**2)+np.sum(self.W1**2))
        return J



    def costFunctionPrime(self,X,y):
        #compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)
        delta3 =  np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdw2 = np.dot(self.a2.T, delta3)+ self.lambduh*self.W2
        delta2 = np.dot(delta3, self.W2.T)* self.sigmoidPrime(self.z2)
        dJdw1 = np.dot(X.T, delta2)+ self.lambduh*self.W1
        return  dJdw1, dJdw2

    def ComputeGradients(self, X,y):
        dJdW1, dJdW2 = self.costFunctionPrime(X,y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



a1 = np.array([[3,3]])
a2 = np.array([[3,-3]])
X = arr = np.random.standard_normal((10, 2))
Z = np.random.standard_normal(10)
# print X,Z
X = np.array([[2,3],[4,5],[9,3]])
y = np.array([[.75],[.80],[.93]])
print np.shape(a1.T)
# y=np.dot(X,a1.T) + np.dot(X,a2.T)**2 + np.multiply(.3,Z)
# print y
# print np.shape(a1*X)
# print np.shape(a2*X)
scalar =.3
NN= Neural_Network()
# for i in range(len(y)):
#     cost1 = NN.costFunction(X[i],y[i])
#     dJdW1, dJdW2 = NN.costFunctionPrime(X[i],y[i])
#     NN.W1 = NN.W1[i] - scalar * dJdW1
#     NN.W2 = NN.W2[i] - scalar * dJdW2

# numgrad = computeNumericalGradient(NN, X,y)
grad = NN.ComputeGradients(X,y)
# print numgrad
# print grad

cost1 = NN.costFunction(X,y)

dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
scalar = 3
NN.W1 = NN.W1 + scalar * dJdW1
NN.W2 = NN.W2 + scalar * dJdW2
cost2 = NN.costFunction(X,y)

dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
NN.W1 = NN.W1 - scalar * dJdW1
NN.W2 = NN.W2 - scalar * dJdW2
cost3 = NN.costFunction(X,y)
print cost1, cost2, cost3
# T = trainer(NN)
# T.train(X,y)
# plt.plot(T.J)
# plt.grid(1)
# plt.xlabel('Iterations')
# plt.ylabel('Cost')



# print cst, dJdW1, dJdW2, NN.W1





