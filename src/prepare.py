import os, numpy as np
from tensorflow import keras

os.makedirs("data/raw", exist_ok=True)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
np.savez_compressed("data/raw/fashion_mnist.npz",
                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)