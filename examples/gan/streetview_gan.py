import numpy as np
import matplotlib.pyplot as plt

from gan import GAN

if __name__ == '__main__':
    X = np.loadtxt('data/streetview/front_streetview.csv', delimiter=',', dtype=np.float64)
    X = (X - 127.5) / 127.5

    generator_params = {'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999, 'weight_decay': 1e-2, 'clip_value': None}
    discriminator_params = {'lr': 2e-4, 'beta1': 0.5, 'beta2': 0.999, 'weight_decay': 1e-2, 'clip_value': None}
    gan = GAN(X, generator_params, discriminator_params, 'streetview_gan')
    gan.summary()

    try:
        gan.fit(X, epochs=30, batch_size=50)
    except KeyboardInterrupt:
        print('Training interrupted by user. Exiting and generating gif...')
    finally:
        gan._generate_training_gif()
