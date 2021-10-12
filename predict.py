import argparse

import tensorflow as tf
import numpy as np
import data_handler as dh
import sys
import matplotlib.pyplot as plt
import keras
from decimal import *

# HELPER FUNCTIONS
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def init_weights(shape):
    """Returns random initial weights"""
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    """Returns random initial biases"""
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    """Returns a 2d convolution operation with stride size 1 and padding SAME"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    """Returns a 2 by 2 pooling operation with padding SAME"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape, name="unspecified"):
    """Returns a convolutional layer with random weights and biases"""
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = init_weights(shape)
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = init_bias([shape[3]])
        with tf.name_scope("Wx_plus_b"):
            preactive = conv2d(input_x, W) + b
            tf.summary.histogram("pre_activations", preactive)
    activations = tf.nn.relu(preactive, name="activation")
    tf.summary.histogram("activations", activations)
    return activations


def normal_full_layer(input_layer, size, act=tf.nn.relu, name="unspecified"):
    """Returns a full layer with random weights and biases"""
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        with tf.name_scope("weights"):
            W = init_weights([input_size, size])
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = init_bias([size])
            variable_summaries(b)
        with tf.name_scope("Wx_plus_b"):
            preactive = tf.matmul(input_layer, W) + b
            tf.summary.histogram("pre_activations", preactive)
        activations = act(preactive, name="activation")
        tf.summary.histogram("activations", activations)
        return activations


def predict(single_image):
    X = np.array([single_image])  # Create array from image to fit shape of x (?,32,32,1)
    checkpoint = "models/first_try.h5"  # Model used for prediction, must have the same graph structure!

    # DICT
    classes = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "A",
        11: "B",
        12: "C",
        13: "D",
        14: "E",
        15: "F",
        16: "G",
        17: "H",
        18: "I",
        19: "J",
        20: "K",
        21: "L",
        22: "M",
        23: "N",
        24: "O",
        25: "P",
        26: "Q",
        27: "R",
        28: "S",
        29: "T",
        30: "U",
        31: "V",
        32: "W",
        33: "X",
        34: "Y",
        35: "Z",
        36: "a",
        37: "b",
        38: "d",
        39: "e",
        40: "f",
        41: "g",
        42: "h",
        43: "n",
        44: "q",
        45: "r",
        46: "t"
    }

    tf.keras.backend.clear_session()
    model = keras.models.load_model("models/first_try.h5")
    model.load_weights('models/first_try_weights5')
    model.summary()
    # correct output somehow
    predictions = model.predict(X)

    return classes[predictions.argmax()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="Path to the image file")
    args = vars(ap.parse_args())
    single_image = dh.get_2d_array(args["image"])
    print("\nResult: \"" + predict(single_image) + "\".")
