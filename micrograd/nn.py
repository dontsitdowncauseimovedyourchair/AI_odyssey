import random
from engine import Value


class Module:
    def zero_grad(self):
        pass

    def parameters(self):
        pass


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        pass

    def __call__(self, x):
        pass

    def parameters(self):
        pass
