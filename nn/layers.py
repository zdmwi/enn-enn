import numpy as np
from abc import ABC, abstractmethod

from nn.utils import xavier_initialization, kaiming_initialization


class Layer(ABC):
    @abstractmethod
    def __init__(self):
        self.__prevIn = np.zeros(0)
        self.__prevOut = np.zeros(0)

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, dataOut):
        self.__prevOut = dataOut

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward(self, gradIn):
        pass


class InputLayer(Layer):
    def __init__(self, dataIn, epsilon=1e-8):
        super(Layer, self).__init__()
        self.epsilon = epsilon
        self.setPrevIn(dataIn)

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = dataIn
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        pass

    def backward(self):
        pass


class BatchNormalizationLayer(Layer):
    """Implementation of batch normalization based on https://arxiv.org/pdf/1502.03167.pdf"""
    id = 0

    def __init__(self, momentum=0.9):
        super(Layer, self).__init__()
        self.name = "BatchNormalizationLayer" + str(BatchNormalizationLayer.id)
        BatchNormalizationLayer.id += 1

        self.epsilon = 1e-8
        self.momentum = momentum
        # we set gamma and beta to 1 and 0 respectively. Batch normalization takes our input
        # and tries to keep it unit gaussian. Before any internal convariance shifts occur, the
        # regular xhat calculation will already be unit gaussian so since there is no shift yet we
        # safely use these as initial values
        self.beta = 0
        self.gamma = 1
        self.mu = None
        self.var = None
        self.xhat = None

        self.empirical_mean = None
        self.empirical_var = None

    def forward(self, dataIn, training=True):
        # calculate the mini-batch mean and variance
        if not training:
            return self.gamma * (dataIn - self.empirical_mean) / np.sqrt(self.empirical_var + self.epsilon) + self.beta

        self.setPrevIn(dataIn)

        self.mu = np.mean(dataIn, axis=0)
        self.var = np.var(dataIn, axis=0)

        # normalize the data
        self.xhat = (dataIn - self.mu) / np.sqrt(self.var + self.epsilon)

        # scale and shift
        Y = self.gamma * self.xhat + self.beta
        self.setPrevOut(Y)

        return Y

    def updateParams(self, eta=1e-4):
        N = self.getPrevIn().shape[0]

        dJdgamma = np.sum(self.getPrevOut()[:N] * self.xhat, axis=0)
        dJdbeta = np.sum(self.getPrevOut()[:N], axis=0)

        self.gamma -= eta*dJdgamma
        self.beta -= eta*dJdbeta

        # keep track of the moving averages of the mean and variance
        if self.empirical_mean is None:
            self.empirical_mean = self.mu
            self.empirical_var = self.var
        else:
            self.empirical_mean = self.momentum * self.empirical_mean + (1 - self.momentum) * self.mu
            self.empirical_var = self.momentum * self.empirical_var + (1 - self.momentum) * self.var
        
    def gradient(self):
        dJdXhat = self.getPrevOut() * self.gamma
        dJdvar = np.sum(dJdXhat * (self.getPrevIn() - self.mu) * -0.5 * np.power(self.var + self.epsilon, -1.5), axis=0)
        dJdmu = np.sum(dJdXhat * -1 / np.sqrt(self.var + self.epsilon), axis=0) + dJdvar * np.mean(-2 * (self.getPrevIn() - self.mu), axis=0)
        dJdX = dJdXhat * 1 / np.sqrt(self.var + self.epsilon) + dJdvar * 2 * (self.getPrevIn() - self.mu) / self.getPrevIn().shape[0] + dJdmu / self.getPrevIn().shape[0]

        return dJdX

    def backward(self, gradIn):
        return gradIn * self.gradient()[0] # we only backpropagate the gradient with respect to the input


class DropoutLayer(Layer):
    def __init__(self, p):
        super(Layer, self).__init__()

        self.p = p
        self.mask = None

    def forward(self, dataIn, training=True):
        self.setPrevIn(dataIn)
        Y = np.array(dataIn)
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size=Y.shape) * (1 / (1- self.p))
            Y *= self.mask
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        return self.getPrevOut() * self.mask

    def backward(self, gradIn):
        return gradIn * self.gradient()


class ActivationLayer(Layer):
    def __init__(self):
        super(Layer, self).__init__()

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass


    @abstractmethod
    def backward(self, gradIn):
        pass


class LinearLayer(ActivationLayer):
    def __init__(self):
        super(ActivationLayer, self).__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = dataIn # for illustration purposes
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        """Returns an N x K tensor"""

        N = self.getPrevIn().shape[0]
        K = self.getPrevIn().shape[1]

        tensor = []
        for _ in range(N):
            jacobian = np.zeros(K)
            for j in range(K):
                jacobian[j] = 1
            
            tensor.append(jacobian)

        return np.stack(tensor)

    def backward(self, gradIn):
        return gradIn * self.gradient()


class ReLuLayer(ActivationLayer):
    def __init__(self):
        super(ActivationLayer, self).__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.maximum(0, dataIn)
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        """Returns an N x K matrix"""

        N = self.getPrevIn().shape[0]
        K = self.getPrevOut().shape[1]
        
        tensor = []
        for obs_idx in range(N):
            jacobian = np.zeros(K)
            for j in range(K):
                jacobian[j] = np.maximum(0, 1 if self.getPrevIn()[obs_idx,j] >= 0 else 0)

            tensor.append(jacobian)

        return np.stack(tensor)

    def backward(self, gradIn):
        return gradIn * self.gradient()


class LogisticSigmoidLayer(ActivationLayer):
    def __init__(self):
        super(ActivationLayer, self).__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.where(
            dataIn >= 0, 
            1.0 / (1.0 + np.exp(-dataIn, dtype=np.float64)), 
            np.exp(dataIn) / (1.0 + np.exp(dataIn, dtype=np.float64)))
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        """Returns an N x K matrix"""

        N = self.getPrevOut().shape[0]
        K = self.getPrevOut().shape[1]

        tensor = []
        for obs_idx in range(N):
            jacobian = np.zeros(K)
            for j in range(K):
                jacobian[j] = self.getPrevOut()[obs_idx, j] * (1 - self.getPrevOut()[obs_idx,j]) 

            tensor.append(jacobian)

        return np.stack(tensor)

    def backward(self, gradIn):
        return gradIn * self.gradient()


class SoftmaxLayer(ActivationLayer):
    def __init__(self):
        super(ActivationLayer, self).__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
    
        max_x = np.amax(dataIn, axis=1, keepdims=True)
        exp = np.exp(dataIn - max_x)
        Y = exp / np.sum(exp, axis=1, keepdims=True)

        self.setPrevOut(Y)

        return Y

    def gradient(self):
        """Returns an N x (K x K) tensor"""

        N = self.getPrevOut().shape[0]
        K = self.getPrevOut().shape[1]

        tensor = []
        for obs_idx in range(N):
            jacobian = np.zeros((K, K))
            for j in range(K):
                for i in range(K):
                    jacobian[j, i] = self.getPrevOut()[obs_idx, i] * ((i == j) - self.getPrevOut()[obs_idx, j])
                
            tensor.append(jacobian)

        return np.stack(tensor)


    def backward(self, gradIn):
        m = []
        grad = self.gradient()
        N = grad.shape[0]
        for i in range(N):
            product = grad[i].dot(gradIn[i])
            m.append(product)

        return np.array(m)


class TanhLayer(ActivationLayer):
    def __init__(self):
        super(ActivationLayer, self).__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        Y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))

        self.setPrevOut(Y)
        
        return Y

    def gradient(self):
        """Returns an N x K matrix"""

        N = self.getPrevOut().shape[0]
        K = self.getPrevOut().shape[1]

        tensor = []
        for obs_idx in range(N):
            jacobian = np.zeros(K)
            for j in range(K):
                jacobian[j] = 1 - self.getPrevOut()[obs_idx,j] ** 2

            tensor.append(jacobian)

        return np.stack(tensor)

    def backward(self, gradIn):
        return gradIn * self.gradient()


class FullyConnectedLayer(Layer):
    id = 0

    def __init__(self, sizeIn, sizeOut, weight_init=None):
        super(Layer, self).__init__()
        self.__sizeIn = sizeIn
        self.__sizeOut = sizeOut

        self.name = "FullyConnectedLayer_" + str(FullyConnectedLayer.id)
        FullyConnectedLayer.id += 1

        epsilon = 1e-4

        # initialize weights and biases
        if weight_init == 'xavier':
            weights, biases = xavier_initialization(sizeIn, sizeOut)
            self.__weights = weights
            self.__biases = biases
        elif weight_init == 'kaiming':
            weights, biases = kaiming_initialization(sizeIn, sizeOut)
            self.__weights = weights
            self.__biases = biases
        else:
            self.__weights = np.random.uniform(-epsilon, epsilon, (sizeIn, sizeOut))
            self.__biases = np.random.uniform(-epsilon, epsilon, (1, sizeOut))

    def getSizeIn(self):
        return self.__sizeIn

    def getSizeOut(self):
        return self.__sizeOut

    def getWeights(self):
        return self.__weights

    def setWeights(self, weights):
        self.__weights = weights
        
    def updateWeights(self, gradIn, eta=1e-4, weight_decay=0.0):
        N = self.getPrevIn().shape[0]

        dJdb = np.sum(gradIn, axis=0) / N
        dJdW = (self.getPrevIn()[:N].T.dot(gradIn[:N])) / N

        self.__weights -= eta * (weight_decay * self.getWeights() + dJdW)
        self.__biases -= eta * dJdb

    def getBiases(self):
        return self.__biases

    def setBiases(self, biases):
        self.__biases = biases

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.array(dataIn.dot(self.getWeights()) + self.getBiases(), dtype=np.float64)
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        return self.getWeights().T

    def backward(self, gradIn):
        return gradIn.dot(self.gradient())


class LeakyReLuLayer(ActivationLayer):
    def __init__(self, alpha=0.3):
        super(ActivationLayer, self).__init__()
        self.__alpha = alpha

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.maximum(0, dataIn) + self.__alpha * np.minimum(0, dataIn)
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        """Returns an N x K matrix"""

        N = self.getPrevIn().shape[0]
        K = self.getPrevOut().shape[1]
        
        tensor = []
        for obs_idx in range(N):
            jacobian = np.zeros(K)
            for j in range(K):
                jacobian[j] = np.maximum(0, 1 if self.getPrevIn()[obs_idx,j] >= 0 else 0) + self.__alpha * np.minimum(0, 1 if self.getPrevIn()[obs_idx,j] < 0 else 0)

            tensor.append(jacobian)

        return np.stack(tensor)

    def backward(self, gradIn):
        return gradIn * self.gradient()


class FlattenLayer(Layer):
    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.reshape(dataIn, -1)
        self.setPrevOut(Y)

        return Y

    def gradient(self):
        return np.eye(self.getPrevIn().shape[1])

    def backward(self, gradIn):
        return np.reshape(gradIn, self.gradient(), order='F')