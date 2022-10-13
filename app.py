'''
Website example for personal study: https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/

pip freeze > requirements.txt # criar a lista de bibliotecas 
pip install -r requirements.txt # baixar
'''

# Library necessary
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


'''
Construção do modelo:
*** Quanto menor a dimensão, maior será a compressão
'''
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