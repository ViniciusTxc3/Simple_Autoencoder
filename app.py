'''
Website example for personal study: https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/

pip freeze > requirements.txt # criar a lista de bibliotecas 
pip install -r requirements.txt # baixar
'''

# Library necessary
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from pickletools import optimize
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np

# '''
# Construção do modelo:
# *** Quanto menor a dimensão, maior será a compressão
# '''
def Simple_Model_Autoencoder():
    encoding_dim = 15 # numero de dimensoes que decideo quanto a entrada será compactada
    input_img = Input(shape=(784,))
    # Representação codificada
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # Representação decodificada
    decoded = Dense(784, activation='sigmoid')(encoded)
    # Modelo que pega a imagem de entrada e motra a decodificação da imagem
    autoencoder = Model(input_img, decoded)

    '''
    Construção do modelo do codificador e decodificador separados para facilitar 
    a diferença entre a entrada e a saída
    '''
    # Modelo de imagens codificadas
    encoder = Model(input_img, encoded)
    # Criação do modelo decodificado
    encoded_input = Input(shape=(encoding_dim,))
    # Ultima camada do modelo autoencoder
    decoder_layer = autoencoder.layers[-1]
    # Modelo decoder
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    '''
    Compilando modelo com o otimizador ADAM
    E ajuste da função de perda de entropia cruzada
    '''
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Carregando os dados
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    # Ver como os dados estão em plot
    plt.imshow(x_train[0].reshape(28,28))
    # plt.show()

    # Treinando o sinal
    autoencoder.fit(x_train, x_train,
                    epochs=15,
                    batch_size=256,
                    validation_data=(x_test,x_test))

    # Fornecer entrada e plotar resultados
    encoded_img = encoder.predict(x_test)
    decoded_img = decoder.predict(encoded_img)
    plt.figure(figsize=(20,4))
    for i in range(5):
        # Display original
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

#####################################################################

'''
# Deep CNN Autoencoder:

# Como estamos usando imagens, faz sentido usar uma rede neural convolucional (CNN).
# Composto por um vetor de camadas Conv2D e max-pooling e o decodificador será um vetor Conv2D e Upsampling.
# '''
def Deep_CNN_Autoencoder():
    model = keras.Sequential()
    # encoder network
    model.add(Conv2D(30, 3, activation='relu', padding='same', input_shape = (28,28,1)))
    model.add(MaxPool2D(2, padding = 'same'))
    model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    model.add(MaxPool2D(2, padding = 'same'))
    # decoder network
    model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(30, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(1, 3, activation= 'sigmoid', padding='same')) # output layer
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy')
    model.summary()

    # Carregando dados e treinando o modelo
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float')/255.
    x_test = x_test.astype('float')/255.
    x_train = np.reshape(x_train,  (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test,  (len(x_test), 28, 28, 1))
    model.fit(x_train, x_train,
                    epochs=15,
                    batch_size=128,
                    validation_data=(x_test, x_test))

    # Fornecendo o input e gerando a saída dos resultados
    pred = model.predict(x_test)
    plt.figure(figsize=(20, 4))
    for i in range(5):
        # Display original
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Display reconstruction
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(pred[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# #####################################################################

'''
Utilizando o modelo para ver o comportamento em imagens ruidosas, alterando cor e insrindo ruídos

Denoising Autoencoder:
'''
def Denoising_Autoencoder():
    # Carregando dados e treinando o modelo
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float')/255.
    x_test = x_test.astype('float')/255.
    x_train = np.reshape(x_train,  (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test,  (len(x_test), 28, 28, 1))


    noise_factor = 0.7
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    # Plot figuras ruidosas
    plt.figure(figsize=(20, 2))
    for i in range(1, 5 + 1):
        ax = plt.subplot(1, 5, i)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    '''
    Com as imagens ruidosas, vamos aplicar a técnica autoencoder e modificando as camadas do modelo para aumentar 
    o filtro para que o modelo tenha um melhor desempenho e depois se ajuste ao modelo.'''
    model = keras.Sequential()
    # encoder network
    model.add(Conv2D(35, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))
    model.add(MaxPool2D(2, padding= 'same'))
    model.add(Conv2D(25, 3, activation= 'relu', padding='same'))
    model.add(MaxPool2D(2, padding= 'same'))
    #decoder network
    model.add(Conv2D(25, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(35, 3, activation= 'relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(1,3,activation='sigmoid', padding= 'same')) # output layer
    model.compile(optimizer= 'adam', loss = 'binary_crossentropy')
    model.fit(x_train_noisy, x_train,
                    epochs=2,
                    batch_size=128,
                    validation_data=(x_test_noisy, x_test))

    # Depois do treinamento, plot dos resultados finais
    pred = model.predict(x_test_noisy)
    plt.figure(figsize=(20, 4))
    for i in range(5):
        # Display original
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Display reconstruction
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(pred[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    Simple_Model_Autoencoder()
    Deep_CNN_Autoencoder()
    Denoising_Autoencoder()
