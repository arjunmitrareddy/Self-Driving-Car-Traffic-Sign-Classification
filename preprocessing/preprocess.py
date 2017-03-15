import numpy as np
import constants as C

def process(X, y):
    X = X.astype('float32')
    X = (X - 128.) / 128.
    y_onehot = np.zeros((y.shape[0], C.N_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.
    y = y_onehot
    return X, y
