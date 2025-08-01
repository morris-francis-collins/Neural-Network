class Scheduler:
    def __init__(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

class ExponentialDecay(Scheduler):
    def __init__(self, lr_0, decay_factor):
        self.lr_0 = lr_0
        self.decay_factor = decay_factor
        self.step_number = 0

    def step(self):
        lr = self.lr_0 * self.decay_factor ** self.step_number
        self.step_number += 1
        return lr