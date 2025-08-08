from config import USE_GPU
if USE_GPU: import cupy as np
else: import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError

    def step(self, lr, parameters, gradients):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self):
        pass

    def step(self, lr, parameters, gradients):
        for i in range(len(parameters)):
            parameters[i] -= lr * gradients[i]

class SGDMomentum(Optimizer):
    def __init__(self, rho=0.9):
        self.velocities = {}
        self.rho = rho
    
    def step(self, lr, parameters, gradients):
        for i in range(len(parameters)):
            parameter = parameters[i]
            parameter_id = id(parameter)

            if parameter_id not in self.velocities:
                # print("parameter shae", i, parameter.shape)
                self.velocities[parameter_id] = np.zeros_like(parameter)

            self.velocities[parameter_id] = self.rho * self.velocities[parameter_id] - lr * gradients[i]
            # print(parameter.shape, i)
            parameter += self.velocities[parameter_id]

class AdaGrad(Optimizer):
    def __init__(self):
        self.v = {}
        self.epsilon = 1e-8
        
    def step(self, lr, parameters, gradients):
        for i in range(len(parameters)):
            parameter = parameters[i]
            parameter_id = id(parameter)

            if parameter_id not in self.v:
                self.v[parameter_id] = np.zeros_like(parameter)

            self.v[parameter_id] += np.square(gradients[i])
            parameter -= lr * gradients[i] / (self.epsilon + np.sqrt(self.v[parameter_id]))

class RMSProp(Optimizer):
    def __init__(self, beta=0.999):
        self.v = {}
        self.beta = beta
        self.epsilon = 1e-8

    def step(self, lr, parameters, gradients):
        for i in range(len(parameters)):
            parameter = parameters[i]
            parameter_id = id(parameter)

            if parameter_id not in self.v:
                self.v[parameter_id] = np.zeros_like(parameter)

            self.v[parameter_id] = self.beta * np.square(gradients[i]) + (1 - self.beta) * np.square(gradients[i])
            parameter -= lr * gradients[i] / (self.epsilon + np.sqrt(self.v[parameter_id]))

class Adam(Optimizer):
    def __init__(self, beta_v=0.999, beta_m=0.9):
        self.v = {}
        self.m = {}
        self.beta_v = beta_v
        self.beta_m = beta_m
        self.t = 1
        self.epsilon = 1e-8
    
    def step(self, lr, parameters, gradients):
        for i in range(len(parameters)):
            parameter = parameters[i]
            parameter_id = id(parameter)

            if parameter_id not in self.v:
                self.v[parameter_id] = np.zeros_like(parameter)
                self.m[parameter_id] = np.zeros_like(parameter)

            self.v[parameter_id] = self.beta_v * self.v[parameter_id] + (1 - self.beta_v) * np.square(gradients[i])
            self.m[parameter_id] = self.beta_m * self.m[parameter_id] + (1 - self.beta_m) * gradients[i]

            v_hat = self.v[parameter_id] / (1 - self.beta_v ** self.t)
            m_hat = self.m[parameter_id] / (1 - self.beta_m ** self.t)

            parameter -= lr * m_hat / (self.epsilon + np.sqrt(v_hat))

        self.t += 1