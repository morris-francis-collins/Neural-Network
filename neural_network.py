class NeuralNetwork: 
    def __init__(self, name, layers, scheduler, optimizer, loss_function):
        self.name = name
        self.layers = layers
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.weights = []
        self.biases = []
        self.loss_function = loss_function

    def train(self, X, Y):
        inference = self.forward_pass(X)
        self.backward_pass(Y)
        return inference
    
    def predict(self, X):
        inference = self.forward_pass(X)
        return inference

    def loss(self, X, Y):
        return self.loss_function.loss(X, Y)

    def forward_pass(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        self.inference = self.loss_function.predict(X)
        return self.inference 
    
    def backward_pass(self, Y):
        delta = self.loss_function.backward(Y)

        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.scheduler, self.optimizer)
