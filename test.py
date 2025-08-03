import numpy as np
import struct
import matplotlib.pyplot as plt
from schedulers import ExponentialDecay
from optimizers import SGD, SGDMomentum, AdaGrad, RMSProp, Adam
from layers import Linear, ReLU, SoftmaxCrossEntropy

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
        np.random.seed(32)

        self.loss_function = SoftmaxCrossEntropy()

    def forward_pass(self, X, Y):
        for layer in self.layers:
            X = layer.forward(X)
        
        inference, loss = self.loss_function.forward(X, Y)
        self.prev_activation = inference
        return inference, loss 
    
    def backward_pass(self, Y):
        delta = self.prev_activation - Y

        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.scheduler, self.optimizer)
            
data = load_mnist_dataset()

print("making")
print(data['train_images'].shape)

layers = [
    Linear(784, 128), 
    ReLU(),
    Linear(128, 32),
    ReLU(),
    Linear(32, 10)
    ]

scheduler = ExponentialDecay(lr_0=0.001, decay_factor=0.99995)

sgd_optimizer = SGD()
# momentum_optimizer = SGDMomentum(layers)
# adagrad_optimizer = AdaGrad(layers)
# rmsprop_optimizer = RMSProp(layers)
# adam_optimizer = Adam(layers)

nn0 = NeuralNetwork("SGD", layers, ExponentialDecay(lr_0=0.01, decay_factor=1.0), sgd_optimizer)
# nn1 = NeuralNetwork("Momentum", layers, ExponentialDecay(lr_0=0.1, decay_factor=0.9999), momentum_optimizer)
# nn2 = NeuralNetwork("AdaGrad", layers, ExponentialDecay(lr_0=0.005, decay_factor=0.99995), adagrad_optimizer)
# nn3 = NeuralNetwork("RMSProp", layers, ExponentialDecay(lr_0=0.0005, decay_factor=0.99995), rmsprop_optimizer)
# nn4 = NeuralNetwork("Adam", layers, ExponentialDecay(lr_0=0.002, decay_factor=0.99995), adam_optimizer) 

neural_networks = [nn0]
accuracies = [[] for _ in range(len(neural_networks))]
training_loss = [[] for _ in range(len(neural_networks))]
test_loss = [[] for _ in range(len(neural_networks))]

epochs = 10
training_size = 6000
test_size = 1000
batch_size = 128

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
        total_loss = 0

        for start in range(0, test_size, batch_size):
            end = min(start + batch_size, test_size)
            effective_batch_size = end - start

            Xb_test = test_images[start : end].reshape(-1, 784).T / 255
            Yb_indices = test_labels[start : end]
            Yb_test = np.eye(10)[Yb_indices].T

            probabilities, loss = nn.forward_pass(Xb_test, Yb_test)
            total_loss += loss * effective_batch_size

            predictions = np.argmax(probabilities, axis=0)
            correct += np.sum(predictions == Yb_indices)

        test_loss[i].append(total_loss / test_size)
        total_loss = 0

        for start in range(0, training_size, batch_size):
            end = min(start + batch_size, training_size)
            effective_batch_size = end - start

            Xb = train_images[start : end].reshape(-1, 784).T / 255.0
            Yb_indices = train_labels[start : end]
            Yb = np.eye(10)[Yb_indices].T
            
            _, loss = nn.forward_pass(Xb, Yb)
            total_loss += loss * effective_batch_size

            nn.backward_pass(Yb)     

        training_loss[i].append(total_loss / training_size)
        accuracies[i].append(100 * correct / test_size)
        print(f"{nn.name}, epoch: {epoch + 1}, accuracy: {100 * correct / test_size}")

epochs = list(range(1, epochs + 1))
plt.figure()
for i in range(len(neural_networks)):
    plt.plot(epochs, accuracies[i], label=neural_networks[i].name)
    plt.plot(epochs, training_loss[i], label="training loss")
    plt.plot(epochs, test_loss[i], label="testing loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()
plt.grid(True)
plt.show()