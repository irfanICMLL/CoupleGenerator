import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


def create_dataset():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    return x_train, y_train, x_test, y_test


def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def setGPU():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = create_dataset()
    model = create_model()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    setGPU()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    keras.set_random_seed(128)
    # model.fit(x_train, y_train, epochs=20, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
