import numpy as np
from scipy import signal

class Layer:
    def __init__(self):
        self.parameters = []

    def forward(self, X):
        raise NotImplementedError

    def backward(self, X, lr, optimizer):
        raise NotImplementedError
        
class Linear(Layer):
    def __init__(self, input_features, output_features):
        self.weights = np.random.randn(input_features, output_features) * np.sqrt(2.0 / input_features)
        self.biases = np.zeros((1, output_features))
        self.parameters = [self.weights, self.biases]

    def forward(self, X):
        self.prev_activation = X
        return X @ self.weights + self.biases

    def backward(self, X, scheduler, optimizer):
        m = X.shape[0]
        dW = (self.prev_activation.T @ X) / m
        db = np.sum(X, axis=0, keepdims=True) / m
        gradients = [dW, db]

        nxt = X @ self.weights.T

        lr = scheduler.step()
        optimizer.step(lr, self.parameters, gradients)

        return nxt
    
class ReLU(Layer):
    def forward(self, X):
        self.prev_z = X
        return np.maximum(0, X)
    
    def backward(self, X, lr, optimizer):
        return X * (self.prev_z > 0)

class Sigmoid(Layer):
    def sigmoid(self, X):
        return np.where(X > 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))
        
    def forward(self, X):
        self.prev_z = X
        return self.sigmoid(X)
    
    def backward(self, X, lr, optimizer):
        sigmoid = self.sigmoid(self.prev_z)
        return X * sigmoid * (1 - sigmoid)
    
class SoftmaxCrossEntropy(Layer):
    def forward(self, X, Y):
        X -= np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X)
        self.probabilities = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        m = X.shape[0]
        log_probabilities = np.log(self.probabilities + 1e-10)
        loss = -np.sum(Y * log_probabilities) / m
    
        return self.probabilities, loss

    def backward(self, Y):
        return self.probabilities - Y
    
class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.biases = np.zeros((output_channels, kernel_size, kernel_size))

    def forward(self, X):
        self.prev = X
        # res = np.zeros(X.shape[0], )

    def backward(self, X):
        pass