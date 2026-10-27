from engine import Value
from nn import MLP


def train():
    """Train an MLP on a toy binary classification dataset."""
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    xy = [1.0, -1.0, -1.0, 1.0]
    nn = MLP(3, [4, 4, 1])
    epochs = 50
    learning_rate = 0.1
    for step in range(epochs):
        loss = sum([(nn(x) - y)**2 for x, y in zip(xs, xy)], Value(0))
        nn.zero_grad()
        loss.backward()
        for p in nn.parameters():
            p.data -= learning_rate * p.grad
        print(f"step {step + 1} - loss: {loss.data}")

    print([nn(x) for x in xs])

train()


