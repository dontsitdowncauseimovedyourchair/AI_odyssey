from engine import Value
from nn import MLP


def train():
    """Train an MLP on a toy binary classification dataset."""
    # TODO: 1. Define a toy dataset of inputs `xs` (e.g. 4 examples of 3 features) and targets `ys` (e.g. [1.0, -1.0, -1.0, 1.0])
    # TODO: 2. Instantiate an MLP model: MLP(3, [4, 4, 1])
    # TODO: 3. Run a training loop for ~20-50 steps:
    #       a) Forward pass: compute predictions for each x in xs, and total squared error loss
    #       b) Zero all parameter gradients
    #       c) Backward pass: backpropagate on the loss
    #       d) Gradient descent: update each parameter's .data using its .grad and a learning rate
    #       e) Print the step number and loss.data
    pass
