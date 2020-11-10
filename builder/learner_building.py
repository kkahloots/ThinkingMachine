import tensorflow as tf
import numpy as np


def make_learner(model_name, input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    #x = tf.keras.layers.Dense(100)(x)
    #x = tf.keras.layers.Dense(100)(x)
    #x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.Dense(units=3)(x)

    x = tf.keras.layers.Flatten()(x)

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    outputs = tf.keras.layers.Dense(units, activation=activation, activity_regularizer=tf.keras.regularizers.l1_l2(l1=1e-3, l2=1e-3))(x)

    learner = tf.keras.Model(name=model_name, inputs=inputs, outputs=outputs)
    print(learner.summary())
    return learner
