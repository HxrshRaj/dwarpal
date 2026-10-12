"""Same forecaster in TensorFlow/Keras, for a like-for-like comparison with the JAX one."""
from __future__ import annotations

import numpy as np


def train(Xtr, ytr, epochs=300, lr=3e-3, hidden=32, seed=0):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    model = tf.keras.Sequential(
        [tf.keras.layers.Input((Xtr.shape[1],)), tf.keras.layers.Dense(hidden, activation="relu"),
         tf.keras.layers.Dense(hidden, activation="relu"), tf.keras.layers.Dense(2)]
    )
    model.compile(tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.Huber(delta=2.0))
    model.fit((Xtr - mu) / sd, ytr, epochs=epochs, batch_size=len(Xtr), verbose=0)
    return lambda Xn: model.predict((Xn - mu) / sd, verbose=0)
