import tensorflow as tf
from sklearn.linear_model import Perceptron
from tensorflow import keras
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(X, y)

