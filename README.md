# Generative-Model-with-CIFAR-10

## Description

This project focuses on building and training a generative model using the CIFAR-10 dataset with Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes, and the goal is to generate new images that resemble the dataset images. This tutorial uses Keras to create and train a generative model such as a Variational Autoencoder (VAE) or Generative Adversarial Network (GAN).

## Dataset

The CIFAR-10 dataset can be directly accessed using Keras, which simplifies the loading and preprocessing of the data.

## Using Google Colab

### 1. Set Up Your Google Colab Environment

- Open Google Colab and create a new notebook.
- Install necessary libraries:

    ```python
    !pip install tensorflow keras
    ```

### 2. Load and Prepare the CIFAR-10 Dataset

- Import and prepare the dataset:

    ```python
    import tensorflow as tf
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import normalize

    # Load the CIFAR-10 dataset
    (x_train, _), (x_test, _) = cifar10.load_data()

    # Normalize the data
    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    # Reshape data to be suitable for the model (if needed)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    ```

### 3. Define the Generative Model

- Define a Variational Autoencoder (VAE) or Generative Adversarial Network (GAN) model using Keras. Here’s an example of a VAE:

    ```python
    from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as K

    # Define the VAE model
    input_img = Input(shape=(32, 32, 3))

    # Encoder
    x = Flatten()(input_img)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    z_mean = Dense(2)(x)
    z_log_var = Dense(2)(x)

    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(256, activation='relu')
    decoder_h_2 = Dense(512, activation='relu')
    decoder_mean = Dense(32*32*3, activation='sigmoid')

    h = decoder_h(z)
    h = decoder_h_2(h)
    decoded_img = Reshape((32, 32, 3))(decoder_mean(h))

    vae = Model(input_img, decoded_img)
    ```

### 4. Compile and Train the Model

- Compile and train the model:

    ```python
    vae.compile(optimizer='adam', loss='binary_crossentropy')

    # Training the VAE
    vae.fit(x_train, x_train,
            epochs=50,
            batch_size=128,
            shuffle=True,
            validation_data=(x_test, x_test))
    ```

### 5. Generate New Images

- Generate new images using the trained model:

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def generate_images(model, num_images=10):
        random_latent_vectors = np.random.normal(size=(num_images, 2))
        generated_images = model.predict(random_latent_vectors)
        return generated_images

    generated_images = generate_images(vae)
    
    # Plot generated images
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))
    for i in range(10):
        ax = axes[i]
        ax.imshow(generated_images[i])
        ax.axis('off')
    plt.show()
    ```

### 6. Save and Share Your Work

- Save your trained model:

    ```python
    vae.save('cifar10_vae.h5')
    ```

- Download the model file:

    ```python
    from google.colab import files
    files.download('cifar10_vae.h5')
    ```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request.

## Contact

mbayandjambealidor@gmail.com
