# Enn-Enn

_"Enn-enn"_ is the phonetic pronunciation of the acronym _"NN"_ which stands for Neural Network.

Enn-enn is a from-scratch deep-learning library heavily influenced by Keras. Built entirely with NumPy, it provides a familiar API for building and training neural networks.

> **Note:** This is an educational project designed for learning how neural networks work under the hood. For production use cases, use established frameworks like PyTorch, TensorFlow or Keras.

## Installation

Requires Python 3.10+

```bash
# Clone the repository
git clone https://github.com/zidanewright/enn-enn.git
cd enn-enn

# Install with uv
uv sync
```

## Quick Start

```python
import numpy as np
from enn_enn.layers import InputLayer, FullyConnectedLayer, ReLuLayer, SoftmaxLayer
from enn_enn.models import Sequential
from enn_enn.optimizers import Adam
from enn_enn.metrics import CrossEntropy, MultiClassAccuracy
from enn_enn.utils import one_hot_encode

# Prepare your data
X_train = ...  # shape: (n_samples, n_features)
y_train = one_hot_encode(labels, n_classes)

# Build a model
model = Sequential([
    InputLayer(X_train),
    FullyConnectedLayer(784, 128, weight_init='kaiming'),
    ReLuLayer(),
    FullyConnectedLayer(128, 64, weight_init='kaiming'),
    ReLuLayer(),
    FullyConnectedLayer(64, 10, weight_init='xavier'),
    SoftmaxLayer()
])

# Compile with optimizer, loss, and metrics
model.compile(
    optimizer=Adam(lr=0.001),
    loss=CrossEntropy(),
    metrics=[MultiClassAccuracy()]
)

# Train
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict
predictions = model.predict(X_test, training=False)
```

## Available Components

### Layers

| Layer | Description |
|-------|-------------|
| `InputLayer` | Input layer for the network |
| `FullyConnectedLayer` | Dense layer with configurable weight initialization |
| `ReLuLayer` | ReLU activation |
| `LeakyReLuLayer` | Leaky ReLU with configurable alpha |
| `LogisticSigmoidLayer` | Sigmoid activation |
| `SoftmaxLayer` | Softmax activation for multi-class output |
| `TanhLayer` | Hyperbolic tangent activation |
| `DropoutLayer` | Dropout regularization |
| `BatchNormalizationLayer` | Batch normalization |
| `FlattenLayer` | Flatten input to 1D |

### Optimizers

| Optimizer | Description |
|-----------|-------------|
| `Adam` | Adam optimizer with support for weight decay and gradient clipping |

### Loss Functions & Metrics

| Class | Description |
|-------|-------------|
| `SquaredError` | Mean squared error (loss & metric) |
| `LogLoss` | Binary cross-entropy (loss & metric) |
| `CrossEntropy` | Categorical cross-entropy (loss & metric) |
| `Accuracy` | Binary accuracy metric |
| `MultiClassAccuracy` | Multi-class accuracy metric |

### Weight Initialization

- `xavier` / Glorot initialization - recommended for tanh/sigmoid activations
- `kaiming` / He initialization - recommended for ReLU activations

## Examples

The `examples/` directory contains GAN implementations trained on MNIST and Street View data:

```bash
# Train a GAN on all MNIST digits
uv run python examples/gan/mnist_all_gan.py

# Train a GAN on digit 2 only
uv run python examples/gan/mnist_digit2_gan.py

# Train a GAN on Street View data
uv run python examples/gan/streetview_gan.py
```

## License

MIT