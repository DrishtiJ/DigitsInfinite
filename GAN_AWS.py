import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist  

# Generator model
def make_generator_model():
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=100, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((28, 28, 1)))
    return model

# Discriminator model
def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
  
# Load and preprocess the dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize images to range [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)

# GAN model
def make_gan_model(generator, discriminator):
    model = models.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

# Initialize models
generator = make_generator_model()
discriminator = make_discriminator_model()
gan = make_gan_model(generator, discriminator)

# Compile models
discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5))
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5))

epochs = 10000
batch_size = 32

for epoch in range(epochs):
  
    # Select a random batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]

    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)

    # Train the discriminator
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print the progress
    print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

    # Save generated images at specified intervals
    if epoch % 100 == 0:
        save_generated_images(epoch, generator)
