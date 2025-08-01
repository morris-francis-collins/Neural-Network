import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError

    def step(self, W, b, dW, db, lr, i):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self):
        pass

    def step(self, W, b, dW, db, lr, i):
        W -= lr * dW
        b -= lr * db

class SGDMomentum(Optimizer):
    def __init__(self, layers, rho=0.9):
        self.weight_velocities = []
        self.bias_velocities = []
        self.rho = rho
    
        for i in range(len(layers) - 1):
            self.weight_velocities.append(np.zeros((layers[i + 1], layers[i])))
            self.bias_velocities.append(np.zeros((layers[i + 1], 1)))
    
    def step(self, W, b, dW, db, lr, i):
        self.weight_velocities[i] = self.rho * self.weight_velocities[i] - lr * dW
        self.bias_velocities[i] = self.rho * self.bias_velocities[i] - lr * db
        W += self.weight_velocities[i]
        b += self.bias_velocities[i]

class AdaGrad(Optimizer):
    def __init__(self, layers):
        self.weight_velocities = []
        self.bias_velocities = []
        self.epsilon = 1e-9
    
        for i in range(len(layers) - 1):
            self.weight_velocities.append(np.zeros((layers[i + 1], layers[i])))
            self.bias_velocities.append(np.zeros((layers[i + 1], 1)))
    
    def step(self, W, b, dW, db, lr, i):
        self.weight_velocities[i] += np.square(dW)
        self.bias_velocities[i] += np.square(db)
        W -= lr * dW / (self.epsilon + np.sqrt(self.weight_velocities[i]))
        b -= lr * db / (self.epsilon + np.sqrt(self.bias_velocities[i]))

class RMSProp(Optimizer):
    def __init__(self, layers, beta=0.99):
        self.weight_velocities = []
        self.bias_velocities = []
        self.beta = beta
        self.epsilon = 1e-9
    
        for i in range(len(layers) - 1):
            self.weight_velocities.append(np.zeros((layers[i + 1], layers[i])))
            self.bias_velocities.append(np.zeros((layers[i + 1], 1)))
    
    def step(self, W, b, dW, db, lr, i):
        self.weight_velocities[i] = self.beta * self.weight_velocities[i] + (1 - self.beta) * np.square(dW)
        self.bias_velocities[i] = self.beta * self.bias_velocities[i] + (1 - self.beta) * np.square(db)

        W -= lr * dW / (self.epsilon + np.sqrt(self.weight_velocities[i]))
        b -= lr * db / (self.epsilon + np.sqrt(self.bias_velocities[i]))

class Adam(Optimizer):
    def __init__(self, layers, beta_m=0.99, beta_v=0.999):
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.t = 1
        self.epsilon = 1e-9

        for i in range(len(layers) - 1):
            self.m_w.append(np.zeros((layers[i + 1], layers[i])))
            self.v_w.append(np.zeros((layers[i + 1], layers[i])))
            self.m_b.append(np.zeros((layers[i + 1], 1)))
            self.v_b.append(np.zeros((layers[i + 1], 1)))
    
    def step(self, W, b, dW, db, lr, i):
        self.m_w[i] = self.beta_m * self.m_w[i] + (1 - self.beta_m) * dW
        self.v_w[i] = self.beta_v * self.v_w[i] + (1 - self.beta_v) * np.square(dW)
        self.m_b[i] = self.beta_m * self.m_b[i] + (1 - self.beta_m) * db
        self.v_b[i] = self.beta_v * self.v_b[i] + (1 - self.beta_v) * np.square(db)

        m_correction = 1 / (1 - self.beta_m ** self.t)
        v_correction = 1 / (1 - self.beta_v ** self.t)
        self.t += 1
    
        W -= lr * m_correction * self.m_w[i] / (self.epsilon + np.sqrt(v_correction * self.v_w[i]))
        b -= lr * m_correction * self.m_b[i] / (self.epsilon + np.sqrt(v_correction * self.v_b[i]))