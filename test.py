import numpy as np
import struct
import matplotlib.pyplot as plt
from schedulers import ExponentialDecay
from optimizers import SGD, SGDMomentum, AdaGrad, RMSProp, Adam

def load_mnist_dataset():
    """Load all MNIST files"""
    files = {
        'train_images': 'data/train-images.idx3-ubyte',
        'train_labels': 'data/train-labels.idx1-ubyte',
        'test_images': 'data/t10k-images.idx3-ubyte',
        'test_labels': 'data/t10k-labels.idx1-ubyte'
    }
    
    data = {}
    
    for key, filename in files.items():
        with open(filename, 'rb') as f:
            if 'images' in key:
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
                data[key] = data[key].reshape(num, rows, cols)
            else:
                magic, num = struct.unpack('>II', f.read(8))
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
    
    return data

class NeuralNetwork: 
    def __init__(self, name, layers, scheduler, optimizer):
        self.name = name
        self.layers = layers
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i + 1], layers[i]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((layers[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
    def forward_pass(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = self.weights[i] @ self.activations[-1] + self.biases[i]
            self.z_values.append(z)

            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)

            self.activations.append(a)

        return self.activations[-1]
    
    def backward_pass(self, Y):
        m = Y.shape[1]
        delta = self.activations[-1] - Y

        for l in range(len(self.weights) - 1, -1, -1):
            A_prev = self.activations[l]

            dW = (delta @ A_prev.T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m

            lr = self.scheduler.step()
            self.optimizer.step(self.weights[l], self.biases[l], dW, db, lr, l)

            if l > 0:
                delta = (self.weights[l].T @ delta) * self.relu_derivative(self.z_values[l-1])
            
data = load_mnist_dataset()

print("making")
print(data['train_images'].shape)

layers = [784, 128, 10]
scheduler = ExponentialDecay(lr_0=0.001, decay_factor=0.99995)

sgd_optimizer = SGD()
momentum_optimizer = SGDMomentum(layers)
adagrad_optimizer = AdaGrad(layers)
rmsprop_optimizer = RMSProp(layers)
adam_optimizer = Adam(layers)

nn0 = NeuralNetwork("SGD", layers, ExponentialDecay(lr_0=0.1, decay_factor=0.9999), sgd_optimizer)
nn1 = NeuralNetwork("Momentum", layers, ExponentialDecay(lr_0=0.1, decay_factor=0.9999), momentum_optimizer)
nn2 = NeuralNetwork("AdaGrad", layers, ExponentialDecay(lr_0=0.005, decay_factor=0.99995), adagrad_optimizer)
nn3 = NeuralNetwork("RMSProp", layers, ExponentialDecay(lr_0=0.0005, decay_factor=0.99995), rmsprop_optimizer)
nn4 = NeuralNetwork("Adam", layers, ExponentialDecay(lr_0=0.002, decay_factor=0.99995), adam_optimizer) 

neural_networks = [nn0, nn1, nn2, nn3, nn4]
accuracies = [[] for _ in range(len(neural_networks))]

epochs = 100
training_size = 10000
test_size = 1000
batch_size = 320

for epoch in range(epochs):
    perm = np.random.permutation(training_size)
    train_images = data['train_images'][perm]
    train_labels = data['train_labels'][perm]

    perm = np.random.permutation(test_size)
    test_images = data['test_images'][perm]
    test_labels = data['test_labels'][perm]

    for i in range(len(neural_networks)):
        nn = neural_networks[i]
    
        correct = 0
        for test in range(test_size):
            ttest = test_images[test].reshape(784, 1) / 255
            ttrue_test = test_labels[test]
            res = nn.forward_pass(ttest)

            if res[ttrue_test][0] == np.max(res):
                correct += 1

        for start in range(0, training_size, batch_size):
            end = start + batch_size
            Xb = train_images[start : end].reshape(-1, 784).T / 255.0
            yb = train_labels[start : end]
            Yb = np.eye(10)[yb].T
            
            _ = nn.forward_pass(Xb)
            nn.backward_pass(Yb)     

        accuracies[i].append(100 * correct / test_size)
        print(f"{nn.name}, epoch: {epoch + 1}, accuracy: {100 * correct / test_size}")


epochs = list(range(1, epochs + 1))
plt.figure()
for i in range(len(neural_networks)):
    plt.plot(epochs, accuracies[i], label=neural_networks[i].name)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()
plt.grid(True)
plt.show()