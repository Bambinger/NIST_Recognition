import numpy as np
import tensorflow as tf
import math
import sys
import time
import datetime
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.regularizers import l1
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import glorot_normal
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NISTHelper():
    def __init__(self, train_img, train_label, test_img, test_label):
        self.i = 0
        self.test_i = 0
        self.training_images = train_img
        self.training_labels = train_label
        self.test_images = test_img
        self.test_labels = test_label

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size]
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def test_batch(self, batch_size):
        x = self.test_images[self.test_i:self.test_i + batch_size]
        y = self.test_labels[self.test_i:self.test_i + batch_size]
        self.test_i = (self.test_i + batch_size) % len(self.test_images)
        return x, y


def unison_shuffled_copies(a, b):
    """Returns 2 unison shuffled copies of array a and b"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]




def log(logstr):
    """Prints logstr to console with current time"""
    print(datetime.datetime.now().isoformat() + " " + logstr)


def main():
    # LOADING DATA
    log("Loading data...")
    images = np.load("nist_images_32x32.npy")
    labels = np.load("nist_labels_32x32.npy")
    log("Data loaded... Shuffling...")
    images, labels = unison_shuffled_copies(images, labels)
    log("Shuffled!")
    split = math.ceil(len(images) * 0.8)
    train_imgs = images[:split]
    train_labels = labels[:split]
    test_imgs = images[split:]
    test_labels = labels[split:]
    log("Performed train-test split")
    nist = NISTHelper(train_imgs, train_labels, test_imgs, test_labels)

    # VARIABLES
    x = Input([32, 32, 1])
    y_true = [47]
    #x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name="x")  # Input, shape = ?x32x32x1
    #y_true = tf.placeholder(tf.float32, shape=[None, 47], name="y_true")  # Labels

    # MODEL

    # filter size=(4,4); channels=1; filters=16; shape=?x32x32x32
    convo_1 = Conv2D(filters = 16, kernel_size = [4,4], padding="same", activation = tf.nn.relu, kernel_initializer='random_normal', bias_initializer='zeros', name="Convolutional_1")(x)
    convo_1_pooling = MaxPool2D(pool_size = [2,2], strides = 2)(convo_1)  # shape=?x16x16x32

    # filter size=(4,4); channels=16; filters=32; shape=?x16x16x64
    convo_2 = Conv2D(filters = 32, kernel_size = [4,4], padding="same", activation = tf.nn.relu, kernel_initializer='random_normal', bias_initializer='zeros', name="Convolutional_2")(convo_1_pooling)
    convo_2_pooling = MaxPool2D(pool_size = [2,2], strides = 2)(convo_2)  # shape=?x8x8x64

    
    # filter size=(4,4); channels=32; filters=64; shape=?x8x8x32
    convo_3 = Conv2D(filters = 64, kernel_size = [4,4], padding="same", activation = tf.nn.relu, kernel_initializer='random_normal', bias_initializer='zeros', name="Convolutional_3")(convo_2_pooling)
    convo_3_pooling = MaxPool2D(pool_size = [2,2], strides = 2)(convo_3)  # shape=4x4x32
    convo_3_flat = Flatten()(convo_3_pooling)  # Flatten convolutional layer

    full_layer_one = Dense(units = 512, activation= tf.nn.relu, kernel_initializer='random_normal', bias_initializer='zeros', name="Normal_Layer_1")(convo_3_flat)
    full_one_dropout = Dropout(rate = 0.3, name = "dropout1")(full_layer_one)
    full_layer_two = Dense(units = 256, activation= tf.nn.relu, kernel_initializer='random_normal', bias_initializer='zeros', name="Normal_Layer_2")(full_one_dropout)
    full_two_dropout = Dropout(rate = 0.2, name = "dropout2")(full_layer_two)
    y_pred = Dense(47, activation = 'softmax', kernel_initializer='random_normal', bias_initializer='zeros', name="Output_Layer")(full_two_dropout)  # Layer with 47 neurons for one-hot encoding

    model = Model(inputs = x, outputs = y_pred)

    model.summary()
    optimizer = Adam(learning_rate=0.005)
    epoch_count = 15
    batch_size = 256
    model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    log("Model created!")



    history = model.fit(x=train_imgs, y=train_labels,
                        validation_data=(test_imgs, test_labels),
                        epochs=epoch_count,
                        batch_size=batch_size)


    log("Finished training.")
    model_path = "models/first_try2.h5"
    model.save("models/first_try2.h5")
    log("Model saved in " + model_path)

    def display_history(history):
        """Summarize history for accuracy and loss.
        """
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()

    display_history(history)

if __name__ == "__main__":
    main()
