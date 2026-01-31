from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

import enn_enn.layers as l
from nn.utils import get_batches


class Model(ABC):
    def __init__(self, layers=[], trainable=True):
        self.metadata = []
        self.layers = []
        for layer in layers:
            self.add(layer)
        self.trainable = trainable

    def summary(self):
        cw = [30, 25, 25] # column widths
        tbl_width = sum(cw)
        print("Model Summary:")
        print("-" * tbl_width)
        print(f"{'Layer (type)'.ljust(cw[0])}{'Output Shape'.ljust(cw[1])}{'# Params'.ljust(cw[1])}")
        print("=" * tbl_width)
        for layer in self.metadata:
            name, output_shape, num_params = layer
            print(f"{name.ljust(cw[0])}{output_shape.ljust(cw[1])}{num_params.ljust(cw[1])}")
            print("-" * tbl_width)

        print(f"Total params: {sum([int(layer[2]) for layer in self.metadata])}")
        print("=" * tbl_width)
        print()
        

    def compile(self, optimizer, loss, metrics=[]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    @abstractmethod
    def add(self, layer):
        pass

    @abstractmethod
    def fit(self, X, Y):
        pass

    def predict(self, X, training=True):
        h = X
        for layer in self.layers:
            if isinstance(layer, l.DropoutLayer) or isinstance(layer, l.BatchNormalizationLayer):
                h = layer.forward(h, training)
            else:
                h = layer.forward(h)
        return h


class Sequential(Model):
    def __init__(self, layers=[], trainable=True):
        self.metadata = []
        self.layers = []
        for layer in layers:
            self.add(layer)
        self.trainable = trainable

    def compile(self, optimizer, loss, metrics=[]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def add(self, layer: l.Layer):
        # track some metadata for debugging purposes
        layer_name = layer.__class__.__name__
        if isinstance(layer, l.FullyConnectedLayer):
            size_in = layer.getSizeIn()
            size_out = layer.getSizeOut()
            output_shape = f"{size_out}"
            num_params = f"{size_in * size_out + size_out}"
            self.metadata.append([layer_name, output_shape, num_params])
        elif isinstance(layer, l.ActivationLayer):
            output_shape = f"{self.metadata[-1][1]}"
            num_params = '0'
            self.metadata.append([layer_name, output_shape, num_params])
        elif isinstance(layer, l.BatchNormalizationLayer):
            output_shape = f"{self.metadata[-1][1]}"
            num_params = f'{int(self.metadata[-1][1]) * 4}'
            self.metadata.append([layer_name, output_shape, num_params])

        self.layers.append(layer)

    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None):
        history = {
            'training_loss': [],
            'validation_loss': [],
            'training_metrics': defaultdict(list),
            'validation_metrics': defaultdict(list)
        }
        for _ in range(epochs):
            # introduce stochasticity by shuffling the data
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
            for X_batch, y_batch in get_batches(X, y, batch_size):
                # we can forward pass on the entire layer list
                # because the loss function is no longer included
                # as part of the model
                h = self.predict(X_batch)

                # do a backward pass on the layers in reverse order
                # and ignore the input layer
                grad = self.loss.gradient(y_batch, h)
                for layer in reversed(self.layers[1:]):
                    newgrad = layer.backward(grad)

                    if isinstance(layer, l.FullyConnectedLayer) and self.trainable:
                        self.optimizer.update(grad, layer)

                    grad = newgrad
            
            # compute training loss and metrics
            h = self.predict(X)
            training_loss = self.loss.eval(y, h)
            history['training_loss'].append(training_loss)
            for metric in self.metrics:
                name = type(metric).__name__
                history['training_metrics'][name].append(metric.eval(y, h))

            # compute validation loss and metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                h = self.predict(X_val)
                validation_loss = self.loss.eval(y_val, h)
                history['validation_loss'].append(validation_loss)

                for metric in self.metrics:
                    name = type(metric).__name__
                    history['validation_metrics'][name].append(metric.eval(y_val, h))
        
        return history
