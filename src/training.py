from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np

from preprocessor import reverseNormPrice, getBaseDf, preprocess, getFeatures

def custom_loss(y_true, y_interval):
    _min = K.min(y_interval, axis=1)
    _max = K.max(y_interval, axis=1)

    bs = tf.shape(y_interval)[0]

    f1 = f2 = f3 = tf.constant(0.0)

    for i in range(bs):
        if y_true[i] < _min[i]:
            f1 += (_min[i] - y_true[i][0]) ** 2

    for i in range(bs):
        if y_true[i] > _max[i]:
            f2 += (y_true[i][0] - _max[i]) ** 2

    f12 = tf.add(f1, f2)

    for i in range(bs):
        f3 += (_max[i] - _min[i]) * 0.05

    return tf.add(f12, f3)

def train_model() -> tf.keras.Model:
    df = getBaseDf()
    df = preprocess(df)

    labels, features = getFeatures(df)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size = 0.1, random_state = 42)
        
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss=custom_loss)

    model.fit(train_features, train_labels, epochs=1, batch_size=32)

    model.evaluate(test_features,  test_labels, verbose=2, batch_size=1)

    predictions = model(test_features).numpy()#.flatten()

    tf.print(tf.shape(predictions[:, 0]))
    
    predictions = reverseNormPrice(predictions, logged=False)
    test_labels = reverseNormPrice(test_labels, logged=False)

    print(predictions[:10])
    print(test_labels[:10])

    errors = abs((0.5 * predictions[:, 0] + 0.5 * predictions[:, 1]) - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'euros.')

    return model

if __name__ == "__main__":
    model = train_model()
    path = '../data/model.h5'
    model.save(path)

