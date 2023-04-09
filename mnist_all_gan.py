import numpy as np
import matplotlib.pyplot as plt

from gan import GAN

if __name__ == '__main__':
    train_data = np.loadtxt('data/mnist/mnist_train.csv', delimiter=',')
    test_data = np.loadtxt('data/mnist/mnist_test.csv', delimiter=',')

    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    X = np.concatenate((X_train, X_test), dtype=np.float64, axis=0)

    X = (X - 127.5) / 127.5

    generator_params = {'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999, 'weight_decay': 0, 'clip_value': 5.0}
    discriminator_params = {'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999, 'weight_decay': 0, 'clip_value': 5.0}
    gan = GAN(X, generator_params, discriminator_params, 'mnist_all_gan')
    gan.summary()

    try:
        gan.fit(X, epochs=30, batch_size=50)
    except KeyboardInterrupt:
        print('Training interrupted by user. Exiting and generating gif...')
    finally:
        gan._generate_training_gif()
