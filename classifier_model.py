# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 04:01:21 2020

@author: Abhinav
"""

from keras.datasets import mnist
from keras import models, layers , optimizers , datasets , utils
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from dataset.game_data import load_data
import labels

def get_data():
    (X_train, Y_train), (X_test, Y_test) = load_data()
    for i in range(len(Y_train)):
        Y_train[i] = labels.label_to_num[Y_train[i]]
    for i in range(len(Y_test)):
        Y_test[i] = labels.label_to_num[Y_test[i]]
    Y_train=utils.to_categorical(Y_train,4)
    Y_test=utils.to_categorical(Y_test,4)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    print(type(X_test))
    return (X_train, Y_train), (X_test, Y_test)



(X_train, Y_train), (X_test, Y_test) = get_data()

print(X_test.shape)
print(Y_test.shape)

# #Neural network architecture
nOutputs = 4

inputs = layers.Input(shape=X_train[0].shape)
cnn = Conv2D(32 , kernel_size=(3,3) , activation='relu')(inputs)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)
cnn = Conv2D(64 , kernel_size=(3,3) , activation='relu')(cnn)
cnn = Dropout(0.25)(cnn)
cnn = Flatten()(cnn)
cnn= Dense(128,activation='relu')(cnn)
cnn = Dropout(0.25)(cnn)
cnn= Dense(32,activation='relu')(cnn)
cnn = Dense(10,activation='relu')(cnn)
outputs = Dense(nOutputs, activation='softmax')(cnn)

model = models.Model ( inputs = inputs, outputs = outputs )
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100)

model.save('brain')

