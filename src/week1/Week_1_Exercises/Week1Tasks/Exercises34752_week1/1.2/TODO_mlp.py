import numpy as np

from perceptron import Perceptron
from activation import ActivationFunction

"""
HINT: Reuse your perceptron.py and activation.py files, and apply the functions directly.
"""


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation: `f(x) = 1/(1+e^(-x))`
    """

    def forward(self, x):
        """
        Activation function output.
        """
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        """
        Activation function derivative.
        """
        return self.forward(x) * (1 - self.forward(x))


class LinearActivation(ActivationFunction):
    """
    Linear activation: `f(x) = x`
    """

    def forward(self, x):
        """
        Activation function output.
        """
        return x

    def gradient(self, x):
        """
        Activation function derivative.
        """
        return np.ones_like(x)  # The gradient of the linear function is 1 everywhere


class Layer:
    def __init__(self, num_inputs, num_units, act_f):
        """
        Initialize the layer, creating `num_units` perceptrons with `num_inputs` each.
        """
        #   TODO Create the perceptrons required for the layer
        self.num_units = num_units
        self.ps = [Perceptron(num_inputs, act_f) for _ in range(num_units)]

    def activation(self, x):
        """Returns the activation `a` of all perceptrons in the layer, given the input vector`x`."""
        return np.array([p.activation(x) for p in self.ps])

    def output(self, a):
        """Returns the output `o` of all perceptrons in the layer, given the activation vector `a`."""
        return np.array([p.output(ai) for p, ai in zip(self.ps, a)])

    def predict(self, x):
        """Returns the output `o` of all perceptrons in the layer, given the input vector `x`."""
        return np.array([p.predict(x) for p in self.ps])

    def gradient(self, a):
        """Returns the gradient of the activation function for all perceptrons in the layer, given the activation vector `a`."""
        return np.array([p.gradient(ai) for p, ai in zip(self.ps, a)])

    def update_weights(self, dw):
        """
        Update the weights of all of the perceptrons in the layer, given the weight change of each.
        Input size: (n_inputs+1, n_units)
        """
        for i in range(self.num_units):
            self.ps[i].w += dw[:, i]

    @property
    def w(self):
        """
        Returns the weights of the neurons in the layer.
        Size: (n_inputs+1, n_units)
        """
        return np.array([p.w for p in self.ps]).T

    def import_weights(self, w):
        """
        Import the weights of all of the perceptrons in the layer.
        Input size: (n_inputs+1, n_units)
        """
        for i in range(self.num_units):
            self.ps[i].w = w[:, i]


class MLP:
    """
       Multi-layer perceptron class

    Parameters
    ----------
    n_inputs : int
       Number of inputs
    n_hidden_units : int
       Number of units in the hidden layer
    n_outputs : int
       Number of outputs
    alpha : float
       Learning rate used for gradient descent
    """

    def __init__(self, num_inputs, n_hidden_units, n_outputs, alpha=1e-3):
        self.num_inputs = num_inputs
        self.n_hidden_units = n_hidden_units
        self.n_outputs = n_outputs

        self.alpha = alpha

        # TODO: Define a hidden layer and the output layer
        self.l1 = Layer(num_inputs, n_hidden_units, Sigmoid)
        self.l_out = Layer(n_hidden_units, n_outputs, LinearActivation)

    def predict(self, x):
        """
        Forward pass prediction given the input x
        TODO: Write the function
        """
        preds = []
        for inp in x:
            # Forward pass through the hidden layer
            a = self.l1.activation(inp)
            a = self.l1.output(a)
            # Forward pass through the output layer
            a = self.l_out.activation(a)
            a = self.l_out.output(a)
            preds.append(a)
        return np.array(preds)

    def train(self, inputs, outputs):
        """
           Train the network

        Parameters
        ----------
        `x` : numpy array
           Inputs (size: n_examples, n_inputs)
        `t` : numpy array
           Targets (size: n_examples, n_outputs)

        TODO: Write the function to iterate through training examples and apply gradient descent to update the neuron weights
        """

        # Loop over training examples
        N = len(inputs)
        dW1    = np.zeros((self.num_inputs + 1, self.n_hidden_units))
        dW_out = np.zeros((self.n_hidden_units + 1, self.n_outputs))
        for inp, target in zip(inputs, outputs):
            # Forward pass
            a1 = self.l1.activation(inp)          # (H,)
            o1 = self.l1.output(a1)               # (H,)
            a2 = self.l_out.activation(o1)        # (O,)
            y  = self.l_out.output(a2)            # (O,)

            # Deltas (use Eq. 7 for sign: δ = f'(a) * (y - t))
            delta_out = self.l_out.gradient(a2) * (y - target)             # (O,)
            delta1    = self.l1.gradient(a1) * (self.l_out.w[1:, :] @ delta_out)  # (H,)

            # Build augmented inputs/outputs to include bias 1
            o0_aug = np.concatenate(([1.0], inp))   # (I+1,)
            o1_aug = np.concatenate(([1.0], o1))    # (H+1,)

            # Accumulate ΔW with shapes that match Layer.update_weights:
            #   ΔW1: (I+1, H) ;  ΔWout: (H+1, O)
            dW1      += np.outer(o0_aug, delta1)        # (I+1, H)
            dW_out   += np.outer(o1_aug, delta_out)     # (H+1, O)


        # Update weights
        self.l1.update_weights(-self.alpha / N * dW1)
        self.l_out.update_weights(-self.alpha / N * dW_out)
            # Reset temporary arrays
        return None  # remove this line

    def export_weights(self):
        return [self.l1.w, self.l_out.w]

    def import_weights(self, ws):
        if ws[0].shape == (self.l1.n_units, self.n_inputs + 1) and ws[1].shape == (
            self.l2.n_units,
            self.l1.n_units + 1,
        ):
            print("Importing weights..")
            self.l1.import_weights(ws[0])
            self.l2.import_weights(ws[1])
        else:
            print("Sizes do not match")


def calc_prediction_error(model, x, t):
    """Calculate the average prediction error"""
    return np.mean((model.predict(x) - t)**2)




if __name__ == "__main__":
    # TODO: Training data
    data = np.array(
        [
            [0.5, 0.5, 0],
            [1.0, 0, 0],
            [2.0, 3.0, 0],
            [0, 1.0, 1],
            [0, 2.0, 1],
            [1.0, 2.2, 1],
        ]
    )
    xdata = data[:, :2]
    ydata = data[:, 2]
    print(xdata)
    print(ydata)

    # TODO: Initialization
    model = MLP(num_inputs=2, n_hidden_units=3, n_outputs=1)

    # TODO: Write a for loop to train the network for a number of iterations. Make plots.
    num_iterations = 1000
    for i in range(num_iterations):
        model.train(xdata, ydata)
        if i % 100 == 0:
            error = calc_prediction_error(model, xdata, ydata)
            print(f"Iteration {i}, Error: {error}")
            print(f"Weights: {model.export_weights()}")
    print("Training complete.")
    

