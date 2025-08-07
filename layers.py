import numpy as np
from scipy import signal
import math 
from numpy.lib.stride_tricks import as_strided
np.random.seed(322)

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
    
    def backward(self, X, scheduler, optimizer):
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

class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, X, lr, optimizer):
        return X.reshape(self.input_shape)
    
class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        self.biases = np.zeros((self.output_channels, 28, 28))
        self.parameters = [self.weights, self.biases]
        
    def forward(self, X):
        self.prev = X
        m, _, height, width = X.shape
        output_height = height + 2 * self.padding - self.kernel_size + 1
        output_width = width + 2 * self.padding - self.kernel_size + 1
        output = np.zeros((m, self.output_channels, output_height, output_width))

        # X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        # shape = (m, self.input_channels, output_height, output_width, self.kernel_size, self.kernel_size)
        # strides = (
        #     X.strides[0],
        #     X.strides[1],
        #     X.strides[2] * 1, # stride = 1
        #     X.strides[3] * 1,
        #     X.strides[2],
        #     X.strides[3]
        # )

        # windows = as_strided(X, shape, strides)

        # for input_channel in range(self.input_channels):
        #     for output_channel in range(self.output_channels):
        #         for i in range(output_height):
        #             for j in range(output_width):
        #                 # print(windows[batch, :, i, j].shape, self.weights[output_channel].shape, output.shape)
        #                 output[:, output_channel, i, j] += np.sum(windows[:, input_channel, i, j] * self.weights[output_channel, input_channel])
        
        # return output + self.biases

        for batch in range(m):
            for output_channel in range(self.output_channels):
                for input_channel in range(self.input_channels):
                    output[batch, output_channel] += signal.correlate2d(
                        np.pad(X[batch, input_channel], self.padding),
                        self.weights[output_channel, input_channel],
                        mode='valid'
                    )
                output[batch, output_channel] += self.biases[output_channel]

        return output
        
    def backward(self, X, scheduler, optimizer):
        m = X.shape[0]
        dW = np.zeros_like(self.weights)
        db = np.sum(X, axis=0) / m
        nxt = np.zeros_like(self.prev) 

        for batch in range(m):
            for output_channel in range(self.output_channels):
                for input_channel in range(self.input_channels):
                    dW[output_channel, input_channel] += signal.correlate2d(self.prev[batch, input_channel], X[batch, output_channel], mode='valid')
                    nxt[batch, input_channel] += signal.convolve2d(np.pad(X[batch, output_channel], self.padding), self.weights[output_channel, input_channel], mode ='valid')

        dW /= m
        lr = scheduler.step()
        optimizer.step(lr, self.parameters, [dW, db])

        return nxt
    
class MaxPool2D(Layer):
    def __init__(self, k=2, stride=None, padding=0):
        super().__init__()
        self.k = k
        self.stride = stride if stride else k
        self.padding = padding

    def forward(self, X):
        self.prev_shape = X.shape
        m, channels, height, width = X.shape
        output_height = (height - self.k) // self.stride + 1
        output_width = (width - self.k) // self.stride + 1

        shape = (m, channels, output_height, output_width, self.k, self.k)
        strides = (
            X.strides[0],
            X.strides[1],
            X.strides[2] * self.stride,
            X.strides[3] * self.stride,
            X.strides[2],
            X.strides[3]
        )
        
        windows = as_strided(X, shape, strides)
        flattened_windows = windows.reshape(m, channels, output_height, output_width, -1)
        output = np.max(flattened_windows, axis=-1)
        flattened_indices = np.argmax(flattened_windows, axis=-1)
        self.indices = np.zeros((m, channels, output_height, output_width, 2), dtype=int)

        for i in range(output_height):
            for j in range(output_width):
                # top-left window corner position
                h_start = i * self.stride
                w_start = j * self.stride

                # position within that window
                h_idx = flattened_indices[:, :, i, j] // self.k 
                w_idx = flattened_indices[:, :, i, j] % self.k

                self.indices[:, :, i, j, 0] = h_start + h_idx
                self.indices[:, :, i, j, 1] = w_start + w_idx
        
        return output
        
    def backward(self, X, scheduler, optimizer):
        m, channels, height, width = self.prev_shape
        output = np.zeros(self.prev_shape)

        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                h_idx = self.indices[:, :, i, j, 0]
                w_idx = self.indices[:, :, i, j, 1]
                batch_indices = np.expand_dims(np.arange(m), axis=-1)
                channel_indices = np.expand_dims(np.arange(channels), axis=0)
                output[batch_indices, channel_indices, h_idx, w_idx] += X[:, :, i, j]
        
        return output