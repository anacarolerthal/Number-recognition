import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from PIL import Image
from numpy import asarray

def predict(filepath):
    image = Image.open(filepath)
    plt.matshow(image)
    image = asarray(image.convert('L'))
    image = image / 255
    image_flattened = image.reshape(1, 28*28)
                                
    prediction = np.argmax(model.predict(image_flattened))

    print(f"O número é {prediction}")


###############################################################################
#                                  Modelo                                     #
###############################################################################

# A linha a seguir salva o dataset em variáveis:
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train e X_train armazenam o array das imagens de treino e teste, onde cada 
# valor da array representa a luminosidade de cada pixel da imagem
# Y_train e y_test armazenam qual número cada imagem representa



# Converte os valores originais das arrays (0 a 255) para um valor entre 0 e 1
X_train = X_train / 255
X_test = X_test / 255

# Converte cada array que originalmente era de um formato 28 por 28 para um
# formato "achatado" com um comprimento de 28*28 (784)
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

# Configura o "formato" do modelo (a quantidade de camadas, a quantidade de
# neurônios em cada camada, entre outras coisas)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(200, activation='sigmoid'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Configura a matemática por trás do modelo, como o tipo de otimização e 
# cálculo de perda
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treina o modelo com o dataset escolhido, e define o total de instâncias
# nas quais o treinamento se repete
model.fit(X_train, y_train, epochs=10)

# Determina o quão preciso o modelo é ao ser aplicado com as imagens de teste
model.evaluate(X_test,y_test)

###############################################################################
#                              Teste de Imagem                                #
###############################################################################

# Inserir imagem manualmente para teste
predict('image.png')






