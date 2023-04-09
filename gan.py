import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import pickle
import time

import nn
import nn.models as m
import nn.layers as l
import nn.optimizers as optimizers
import nn.metrics as metrics
import nn.utils

class GAN:
    def __init__(self, data, gen_params, dis_params, name):
        self.input_shape = data.shape

        self.name = name
        self.gen_params = gen_params
        self.dis_params = dis_params

        # data received is normalized so the mean will be 0 and std will be 1
        self.data_mean = np.mean(data)
        self.data_std = np.std(data)

        self._img_fig = None # used to prevent the figure from being recreated

        # used for generating the gif examples
        n_examples = 1
        self.noise_seed = np.random.normal(self.data_mean, self.data_std, (n_examples, 100))

        # generator setup
        self.generator = self.build_generator(self.noise_seed)
        self.generator.compile(
            optimizer=optimizers.Adam(
                lr=gen_params['lr'], 
                beta1=gen_params['beta1'], 
                beta2=gen_params['beta2'],
                weight_decay=gen_params['weight_decay'],
                clip_value=gen_params['clip_value']),
            loss=metrics.LogLoss()
        )

        # discriminator setup
        self.discriminator = self.build_discriminator(data)
        self.discriminator.compile(
            optimizer=optimizers.Adam(
                lr=dis_params['lr'], 
                beta1=dis_params['beta1'],
                beta2=dis_params['beta2'], 
                weight_decay=dis_params['weight_decay'],
                clip_value=dis_params['clip_value']), 
            loss=metrics.LogLoss()
        )

    def summary(self):
        """Prints a summary of the GAN model"""

        print('Generator:\n')
        self.generator.summary()
        print('Discriminator:\n')
        self.discriminator.summary()

    def build_generator(self, noise):
        """Builds the generator model"""

        model = m.Sequential([
            l.InputLayer(noise),
            l.FullyConnectedLayer(noise.shape[1], 256, weight_init='kaiming'),
            l.LeakyReLuLayer(alpha=0.2),
            l.BatchNormalizationLayer(momentum=0.8),
            l.FullyConnectedLayer(256, 512, weight_init='kaiming'),
            l.LeakyReLuLayer(alpha=0.2),
            l.BatchNormalizationLayer(momentum=0.8),
            l.FullyConnectedLayer(512, 1024, weight_init='kaiming'),
            l.LeakyReLuLayer(alpha=0.2),
            l.BatchNormalizationLayer(momentum=0.8),
            l.FullyConnectedLayer(1024, self.input_shape[1], weight_init='xavier'),
            l.TanhLayer()
        ])

        return model

    def build_discriminator(self, data):
        """Builds the discriminator model"""

        model = m.Sequential([
            l.InputLayer(data),
            l.FullyConnectedLayer(self.input_shape[1], 512, weight_init='kaiming'),
            l.LeakyReLuLayer(alpha=0.2),
            l.FullyConnectedLayer(512, 256, weight_init='kaiming'),
            l.LeakyReLuLayer(alpha=0.2),
            l.FullyConnectedLayer(256, 1, weight_init='xavier'),
            l.LogisticSigmoidLayer()
        ])

        return model

    def fit(self, X, epochs=100, batch_size=100):
        """Trains the GAN model"""

        m = batch_size

        for t in range(epochs):
            start_time = time.time()
            # shuffle the data
            rand_idx = np.random.permutation(X.shape[0])
            X = X[rand_idx]
            for batch in nn.utils.get_batches(X, None, m):
                X_batch = batch[0]

                # 1. Generate m fake samples using the generator
                noise = np.random.normal(self.data_mean, self.data_std, (m, 100))
                Xf = self.generator.predict(noise)

                # 2. Grab m true samples from the batch
                idx = np.random.randint(0, X_batch.shape[0], m)
                Xr = X_batch[idx]

                # 3. Update the weights of the discriminator using the 2m samples

                # Training use the real samples
                hd = self.discriminator.predict(Xr)
                dgrad_r = self.discriminator.loss.gradient(np.ones((m, 1)), hd)
                for layer in reversed(self.discriminator.layers[1:]):
                    dnewgrad_r = layer.backward(dgrad_r)

                    if isinstance(layer, l.FullyConnectedLayer):
                        self.discriminator.optimizer.update(dgrad_r, layer)

                    if isinstance(layer, l.BatchNormalizationLayer):
                        self.discriminator.optimizer.update(dgrad_r, layer)

                    dgrad_r = dnewgrad_r

                derr_r = self.discriminator.loss.eval(np.ones((m, 1)), hd)
        
                # Training use the fake samples
                hd = self.discriminator.predict(Xf)
                dgrad_f = self.discriminator.loss.gradient(np.zeros((m, 1)), hd)
                for layer in reversed(self.discriminator.layers[1:]):
                    dnewgrad_f = layer.backward(dgrad_f)

                    if isinstance(layer, l.FullyConnectedLayer):
                        self.discriminator.optimizer.update(dgrad_f, layer)

                    if isinstance(layer, l.BatchNormalizationLayer):
                        self.discriminator.optimizer.update(dgrad_f, layer)

                    dgrad_f = dnewgrad_f

                derr_f = self.discriminator.loss.eval(np.zeros((m, 1)), hd)

                derr = derr_r + derr_f

                # 4. Update the weights of the generator using the m fake samples
                
                noise = np.random.normal(self.data_mean, self.data_std, (m, 100))
                Xf = self.generator.predict(noise)
                # get the predictions of the updated discriminator
                hd = self.discriminator.predict(Xf)

                # backpropagate the gradient through the discriminator without training
                yg = np.ones((m, 1))
                gerr = self.generator.loss.eval(yg, hd)

                ggrad = self.generator.loss.gradient(yg, hd)
                for layer in reversed(self.discriminator.layers[1:]):
                    ggrad = layer.backward(ggrad)

                # pass the gradient to the generator and backpropagate while updating weights
                for layer in reversed(self.generator.layers[1:]):
                    gnewgrad = layer.backward(ggrad)

                    if isinstance(layer, l.FullyConnectedLayer):
                        self.generator.optimizer.update(ggrad, layer)

                    if isinstance(layer, l.BatchNormalizationLayer):
                        self.generator.optimizer.update(ggrad, layer)

                    ggrad = gnewgrad

            # generate images for gif and print loss
            self._generate_and_save_images(t, self.noise_seed)
            # get number of seconds since start time
            time_taken = time.time() - start_time
            print('Epoch: {0}, Generator Loss: {1}, Discriminator Loss: {2} - Time: {3}s'.format(t, gerr, derr, time_taken))

    def generate(self, n):
        """Generates n samples"""

        noise = np.random.normal(self.data_mean, self.data_std, (n, 100))
        return self.generator.predict(noise)

    def _generate_and_save_images(self, epoch, test_input):
        predictions = 0.5 * self.generator.predict(test_input, training=False) + 0.5 # rescale to [0, 1]

        dims = np.sqrt(self.input_shape[1]).astype(int)

        if self._img_fig is None:
            self._img_fig = plt.figure()

        plt.imshow(predictions[0].reshape(dims, dims), cmap='gray')
        plt.tight_layout()
        plt.axis('off')
        plt.savefig('gif/{}_image_at_epoch_{:04d}.png'.format(self.name, epoch), transparent=True)
        
    def _generate_training_gif(self):
        """Generates a gif of the training process"""
        filename = f'gif/{self.name}.gif'
        with imageio.get_writer(filename, mode='I') as writer:
            filenames = glob.glob(f'gif/{self.name}_image*.png')
            filenames = sorted(filenames)
            for fname in filenames:
                image = imageio.imread(fname)
                writer.append_data(image)
            image = imageio.imread(fname)
            writer.append_data(image)

    def save_model(self, dirname='models'):
        """Saves the model to a file"""
        with open(f'{dirname}/gan.pkl', 'wb') as f:
            pickle.dump(self, f)
