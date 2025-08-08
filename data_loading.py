from config import USE_GPU
if USE_GPU: import cupy as np
else: import numpy as np
import numpy
import os
import struct
import pickle

def load_mnist_dataset():
    files = {
        'train_images': 'data/mnist/train-images.idx3-ubyte',
        'train_labels': 'data/mnist/train-labels.idx1-ubyte',
        'test_images': 'data/mnist/t10k-images.idx3-ubyte',
        'test_labels': 'data/mnist/t10k-labels.idx1-ubyte'
    }
    
    data = {}
    print(f"loading default MNIST")
    
    for key, filename in files.items():
        with open(filename, 'rb') as f:
            if 'images' in key:
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
                data[key] = data[key].reshape(num, rows, cols)
            else:
                magic, num = struct.unpack('>II', f.read(8))
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)

    print(f"dataset size:")
    print(f"train: {data['train_images'].shape}")
    print(f"test: {data['test_images'].shape}")
    
    return data

def load_corrupted_mnist_dataset(corruptions=[]):
    corrupted_path = 'data/corrupted_mnist'
    
    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []
    
    if not corruptions:
        corruptions = [d for d in os.listdir(corrupted_path) 
                    if os.path.isdir(os.path.join(corrupted_path, d))]
    
    print(f"loading {len(corruptions)} corruption types: {corruptions}")
    
    for corruption in corruptions:
        base_path = f'{corrupted_path}/{corruption}'
        all_train_images.append(np.load(f'{base_path}/train_images.npy'))
        all_train_labels.append(np.load(f'{base_path}/train_labels.npy'))
        all_test_images.append(np.load(f'{base_path}/test_images.npy'))
        all_test_labels.append(np.load(f'{base_path}/test_labels.npy'))
    
    data = {
        'train_images': np.concatenate(all_train_images, axis=0),
        'train_labels': np.concatenate(all_train_labels, axis=0),
        'test_images': np.concatenate(all_test_images, axis=0),
        'test_labels': np.concatenate(all_test_labels, axis=0)
    }
    
    print(f"dataset size:")
    print(f"train: {data['train_images'].shape}")
    print(f"test: {data['test_images'].shape}")
    
    return data

def load_cifar10_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    data = batch[b'data']
    labels = batch[b'labels']
    
    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)
    
    return np.array(data), np.array(labels)

def load_cifar10_dataset(data_dir='data/cifar-10'):    
    train_data = []
    train_labels = []

    print(f"loading cifar-10")

    for i in range(1, 6):
        filepath = f'{data_dir}/data_batch_{i}'
        data, labels = load_cifar10_batch(filepath)
        train_data.append(data)
        train_labels.append(labels)
    
    train_data = np.array(numpy.concatenate(train_data, axis=0))
    train_labels = np.array(numpy.concatenate(train_labels, axis=0))
    
    test_data, test_labels = load_cifar10_batch(f'{data_dir}/test_batch')
    
    data = {
        'train_images': train_data,
        'train_labels': train_labels,
        'test_images': test_data,
        'test_labels': test_labels,
    }
    
    print(f"cifar-10 loaded. train: {data['train_images'].shape}, test: {data['test_images'].shape}")
    
    return data