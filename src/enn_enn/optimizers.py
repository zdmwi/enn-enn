from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

import enn_enn.layers as l


class Optimizer(ABC):
    def __init__(self):
        self.lr = 0.001

    @abstractmethod
    def update(self, layer: l.FullyConnectedLayer):
        pass

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0, clip_value=None):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.t = defaultdict(self._init_fn)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.weight_decay = weight_decay

        # both m and v should be the same shape as the weights but we rely
        # on on the update step and broadcasting to do the work for us
        self.m = {}
        self.v = {}

    def _init_fn(self):
        return 1

    def update(self, gradient, layer):
        if layer.name not in self.m:
            self.m[layer.name] = 0
            self.v[layer.name] = 0

        N = gradient.shape[0]

        self.m[layer.name] = self.beta1 * self.m[layer.name] + (1 - self.beta1) * gradient
        self.v[layer.name] = self.beta2 * self.v[layer.name] + (1 - self.beta2) * (gradient ** 2)

        m_hat = self.m[layer.name] / (1 - self.beta1 ** self.t[layer.name])
        v_hat = self.v[layer.name] / (1 - self.beta2 ** self.t[layer.name])

        update = m_hat / (np.sqrt(v_hat) + self.epsilon)
        if self.clip_value is not None:
            update = np.clip(update, -self.clip_value, self.clip_value)
        self.t[layer.name] += 1

        if isinstance(layer, l.FullyConnectedLayer):
            layer.updateWeights(update, eta=self.lr, weight_decay=self.weight_decay)
        elif isinstance(layer, l.BatchNormalizationLayer):
            layer.updateParams(eta=self.lr)