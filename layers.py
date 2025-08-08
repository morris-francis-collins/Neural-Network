from config import USE_GPU
if USE_GPU: 
    import cupy as np
    from cupy.lib.stride_tricks import as_strided
else: 
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
from scipy import signal
# np.random.seed(322)

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
        self.biases = np.zeros((self.output_channels, 32, 32))
        self.parameters = [self.weights, self.biases]
        
    def forward(self, X):
        self.prev_input = X
        m, _, height, width = X.shape
        output_height = height + 2 * self.padding - self.kernel_size + 1
        output_width = width + 2 * self.padding - self.kernel_size + 1

        X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        shape = (m, self.input_channels, output_height, output_width, self.kernel_size, self.kernel_size)
        strides = (
            X.strides[0],
            X.strides[1],
            X.strides[2], # stride = 1
            X.strides[3], # stride = 1
            X.strides[2],
            X.strides[3]
        )

        windows = as_strided(X, shape, strides)
        windows_2d = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            m * output_height * output_width,
            self.input_channels * self.kernel_size * self.kernel_size
        )
        weights_2d = self.weights.reshape(
            self.output_channels,
            self.input_channels * self.kernel_size * self.kernel_size
        )
        
        output_2d = windows_2d @ weights_2d.T
        output = output_2d.reshape(m, output_height, output_width, self.output_channels).transpose(0, 3, 1, 2)

        return output + self.biases        

        # for batch in range(m):
        #     for output_channel in range(self.output_channels):
        #         for input_channel in range(self.input_channels):
        #             output[batch, output_channel] += signal.correlate2d(
        #                 np.pad(X[batch, input_channel], self.padding),
        #                 self.weights[output_channel, input_channel],
        #                 mode='valid'
        #             )
        #         output[batch, output_channel] += self.biases[output_channel]
                
    def backward(self, X, scheduler, optimizer):
        input_height, input_width = self.prev_input.shape[2], self.prev_input.shape[3]
        m, _, output_height, output_width = X.shape
        # dW = np.zeros_like(self.weights)
        # nxt = np.zeros_like(self.prev_input)

        self.prev_input = np.pad(self.prev_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        
        shape = (m, self.input_channels, output_height, output_width, self.kernel_size, self.kernel_size)
        strides = (
            self.prev_input.strides[0],
            self.prev_input.strides[1],
            self.prev_input.strides[2], # stride = 1
            self.prev_input.strides[3], # stride = 1
            self.prev_input.strides[2],
            self.prev_input.strides[3]
        )

        windows = as_strided(self.prev_input, shape, strides)
        windows_2d = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            m * (output_height ) * (output_width ),
            self.input_channels * self.kernel_size * self.kernel_size
        )
        output_grad_2d = X.transpose(0, 2, 3, 1).reshape(
            m * (output_height ) * (output_width), 
            self.output_channels,
        )

        dW_2d = output_grad_2d.T @ windows_2d
        dW = dW_2d.reshape(self.weights.shape) / m
        db = np.sum(X, axis=0) / m

        X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        shape = (m, self.output_channels, input_height, input_width, self.kernel_size, self.kernel_size)
        strides = (
            X.strides[0],
            X.strides[1],
            X.strides[2], # stride = 1
            X.strides[3], # stride = 1
            X.strides[2],
            X.strides[3]
        )

        windows = as_strided(X, shape, strides)
        windows_2d = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            m * input_height * input_width,
            self.output_channels * self.kernel_size * self.kernel_size
        )

        rotated_weights = self.weights[:, :, ::-1, ::-1]
        weights_2d = rotated_weights.transpose(1, 0, 2, 3).reshape(
            self.input_channels,
            self.output_channels * self.kernel_size * self.kernel_size
        )

        grad_input_2d = windows_2d @ weights_2d.T
        grad_input = grad_input_2d.reshape(m, output_height, output_width, self.input_channels).transpose(0, 3, 1, 2)

        # for batch in range(m):
        #     for output_channel in range(self.output_channels):
        #         for input_channel in range(self.input_channels):
        #               dW[output_channel, input_channel] += signal.correlate2d(self.prev[batch, input_channel], X[batch, output_channel], mode='valid')
        #               nxt[batch, input_channel] += signal.convolve2d(np.pad(X[batch, output_channel], self.padding), self.weights[output_channel, input_channel], mode ='valid')

        lr = scheduler.step()
        optimizer.step(lr, self.parameters, [dW, db])

        return grad_input
    
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