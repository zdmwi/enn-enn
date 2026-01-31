import numpy as np
from abc import ABC, abstractmethod


class ObjectiveLayer(ABC):
    @abstractmethod
    def gradient(self):
        pass


class Metric(ABC):
    @abstractmethod
    def eval(self, Y, Yhat):
        pass


class SquaredError(ObjectiveLayer, Metric):
    def eval(self, Y, Yhat):
        loss = np.mean((Y - Yhat) * (Y - Yhat))
        return loss

    def gradient(self, Y, Yhat):
        loss_grad = -2 * (Y - Yhat)
        return loss_grad


class LogLoss(ObjectiveLayer, Metric):
    def eval(self, Y, Yhat, epsilon=1e-4):
        loss = np.mean(-(Y * np.log(Yhat + epsilon) + (1 - Y) * np.log(1 - Yhat + epsilon)))
        return loss

    def gradient(self, Y, Yhat, epsilon=1e-4):
        loss_grad = -(Y - Yhat) / (Yhat * (1 - Yhat) + epsilon)
        return loss_grad


class CrossEntropy(ObjectiveLayer, Metric):
    def eval(self, Y, Yhat, epsilon=1e-4):
        loss = -np.mean(np.sum(Y * np.log(Yhat + epsilon), axis=1))
        return loss

    def gradient(self, Y, Yhat, epsilon=1e-4):
        loss_grad = -Y / (Yhat + epsilon)
        return loss_grad

class Accuracy(Metric):
    def __init__(self, from_logits=False):
        self.from_logits = from_logits

    def eval(self, Y, Yhat):
        if self.from_logits:
            Yhat[Yhat > 0.5] = 1
            Yhat[Yhat <= 0.5] = 0
        return np.mean(Y == Yhat)


class MultiClassAccuracy(Metric):
    def eval(self, Y, Yhat):
        Yhat = np.argmax(Yhat, axis=1)
        Y = np.argmax(Y, axis=1)
        return np.mean(Y == Yhat)