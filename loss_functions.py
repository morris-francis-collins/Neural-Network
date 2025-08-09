from config import USE_GPU
if USE_GPU: import cupy as np
else: import numpy as np

class LossFunction():
    def __init__(self):
        pass

    def predict(self, X):
        return NotImplementedError
    
    def loss(self, X, Y):
        return NotImplementedError

    def backward(self, Y):
        return NotImplementedError

class SoftmaxCrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()
    
    def predict(self, X):
        X -= np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X)
        self.probabilities = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.probabilities

    def loss(self, X, Y):
        m = X.shape[0]
        log_probabilities = np.log(self.probabilities + 1e-10)
        loss = -np.sum(Y * log_probabilities) / m
        return loss
    
    def backward(self, Y):
        return self.probabilities - Y