from config import USE_GPU
if USE_GPU: import cupy as np
else: import numpy as np
import matplotlib.pyplot as plt
import time
from neural_network import NeuralNetwork
from schedulers import ExponentialDecay
from optimizers import SGD, SGDMomentum, AdaGrad, RMSProp, Adam
from data_loading import load_mnist_dataset, load_corrupted_mnist_dataset, load_cifar10_dataset
from layers import Linear, ReLU, Sigmoid, Conv2D, Flatten, MaxPool2D
from loss_functions import SoftmaxCrossEntropy
np.random.seed(32)

print("using gpu" if USE_GPU else "using cpu")
data = load_corrupted_mnist_dataset()
# plt.imshow(data['train_images'][0], cmap='gray'); plt.show()

# layers = [
#     Conv2D(1, 32),
#     ReLU(),
#     Conv2D(16, 64),
#     ReLU(),
#     MaxPool2D(),
#     Flatten(),
#     Linear(64 * 14 * 14, 1024),
#     ReLU(),
#     Linear(1024, 128),
#     ReLU(),
#     Linear(128, 32),
#     ReLU(),
#     Linear(32, 10)
#     ]

# layers = [
#     Conv2D(1, 16),
#     ReLU(),
#     Conv2D(16, 32),
#     ReLU(),
#     MaxPool2D(),
#     Flatten(),
#     Linear(32 * 14 * 14, 256),
#     ReLU(),
#     Linear(256, 10)
# ]

# layers = [
#     Conv2D(3, 32),
#     ReLU(),
#     Conv2D(32, 32),
#     ReLU(),
#     MaxPool2D(),

#     # Conv2D(32, 64),
#     # ReLU(),
#     # Conv2D(64, 64),
#     # ReLU(),
#     # MaxPool2D(),

#     # Flatten(),
#     # Linear(32 * 14 * 14, 256),
#     # ReLU(),
#     # Linear(256, 10)
# ]

layers = [
    Flatten(),
    Linear(784, 256),
    ReLU(),
    Linear(256, 128), 
    ReLU(),
    Linear(128, 32),
    ReLU(),
    Linear(32, 10)
    ]

# layers = [
#     Flatten(),
#     # Linear(784, 4096), 
#     # ReLU(),
#     # Linear(4096, 2048), 
#     # ReLU(),
#     # Linear(784, 1024), 
#     # ReLU(),
#     Linear(784, 512), 
#     ReLU(),
#     Linear(512, 256), 
#     ReLU(),
#     Linear(256, 32),
#     ReLU(),
#     Linear(32, 10)
# ]

scheduler = ExponentialDecay(lr_0=0.001, decay_factor=0.99995)

sgd_optimizer = SGD()
momentum_optimizer = SGDMomentum()
adagrad_optimizer = AdaGrad()
rmsprop_optimizer = RMSProp()
adam_optimizer = Adam()

nn0 = NeuralNetwork("SGD", layers, ExponentialDecay(lr_0=0.01, decay_factor=0.9999), sgd_optimizer, SoftmaxCrossEntropy())
nn1 = NeuralNetwork("Momentum", layers, ExponentialDecay(lr_0=0.01, decay_factor=0.9999), momentum_optimizer, SoftmaxCrossEntropy())
nn2 = NeuralNetwork("AdaGrad", layers, ExponentialDecay(lr_0=0.005, decay_factor=0.99995), adagrad_optimizer, SoftmaxCrossEntropy())
nn3 = NeuralNetwork("RMSProp", layers, ExponentialDecay(lr_0=0.0005, decay_factor=0.99995), rmsprop_optimizer, SoftmaxCrossEntropy())
nn4 = NeuralNetwork("Adam", layers, ExponentialDecay(lr_0=0.001, decay_factor=0.99995), adam_optimizer, SoftmaxCrossEntropy())

neural_networks = [nn4]
accuracies = [[] for _ in range(len(neural_networks))]
training_loss = [[] for _ in range(len(neural_networks))]
test_loss = [[] for _ in range(len(neural_networks))]

epochs = 10
training_size = 50000
test_size = 10000
batch_size = 512

for epoch in range(epochs):
    perm = np.random.choice(data['train_images'].shape[0], training_size)
    train_images = data['train_images'][perm]
    train_labels = data['train_labels'][perm]

    perm = np.random.choice(data['test_images'].shape[0], test_size)
    test_images = data['test_images'][perm]
    test_labels = data['test_labels'][perm]

    for i in range(len(neural_networks)):
        t1 = time.time()
        nn = neural_networks[i]
        correct = 0
        total_loss = 0

        for start in range(0, test_size, batch_size):
            end = min(start + batch_size, test_size)
            effective_batch_size = end - start
            Xb_test = test_images[start : end].reshape(effective_batch_size, -1, 28, 28) / 255
            Yb_indices = test_labels[start : end]
            Yb_test = np.eye(10)[Yb_indices]

            probabilities = nn.predict(Xb_test)
            loss = nn.loss(Xb_test, Yb_test)
            total_loss += loss * effective_batch_size

            predictions = np.argmax(probabilities, axis=1)
            correct += np.sum(predictions == Yb_indices)
            
        test_loss[i].append(total_loss / test_size)
        total_loss = 0

        for start in range(0, training_size, batch_size):
            end = min(start + batch_size, training_size)
            effective_batch_size = end - start

            Xb = train_images[start : end].reshape(effective_batch_size, -1, 28, 28) / 255
            Yb_indices = train_labels[start : end]
            Yb = np.eye(10)[Yb_indices]
            
            probabilities = nn.train(Xb, Yb)
            loss = nn.loss(Xb, Yb)
            total_loss += loss * effective_batch_size

        training_loss[i].append(total_loss / training_size)
        accuracies[i].append(100 * correct / test_size)

        print(f"{nn.name}, epoch: {epoch + 1}, accuracy: {100 * correct / test_size}, time: {round(time.time() - t1, 2)}")

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