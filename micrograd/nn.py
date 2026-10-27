import random
from engine import Value


class Module:
    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        out = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        return out.tanh() if self.nonlin else out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = []
        for neuron in self.neurons:
            out.append(neuron(x))
        return out[0] if len(out) == 1 else out


    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        layersz = [nin] + nouts
        self.layers = [Layer(layersz[i], layersz[i+1]) for i in range(len(layersz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

