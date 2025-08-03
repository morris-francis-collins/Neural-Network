import numpy as np

class Layer:
    def __init__(self):
        self.parameters = []

    def forward(self, X):
        raise NotImplementedError

    def backward(self, X):
        raise NotImplementedError

    def update_parameters(self):
        raise NotImplementedError
    
class Linear(Layer):
    def __init__(self, input_features, output_features):        
        self.weights = np.random.randn(output_features, input_features) * np.sqrt(2.0 / input_features)
        self.biases = np.zeros((output_features, 1))
        self.parameters = [self.weights, self.biases]

    def forward(self, X):
        self.prev_activation = X
        return self.weights @ X + self.biases

    def backward(self, X, scheduler, optimizer):
        m = X.shape[1]
        dW = (X @ self.prev_activation.T) / m
        db = np.sum(X, axis=1, keepdims=True) / m
        gradients = [dW, db]

        nxt = self.weights.T @ X

        lr = scheduler.step()
        optimizer.step(lr, self.parameters, gradients)

        return nxt
    
class ReLU(Layer):
    def forward(self, X):
        self.prev_z = X
        return np.maximum(0, X)
    
    def backward(self, X, lr, optimizer):
        return X * (self.prev_z > 0)
    
class SoftmaxCrossEntropy(Layer):
    def forward(self, X, Y):
        X -= np.max(X, axis=0, keepdims=True)
        exp_X = np.exp(X)
        self.probabilities = exp_X / np.sum(exp_X, axis=0, keepdims=True)

        m = X.shape[1]
        log_probabilities = np.log(self.probabilities + 1e-10)
        loss = -np.sum(Y * log_probabilities) / m
    
        return self.probabilities, loss

    def backward(self, Y):
        return self.probabilities - Y
