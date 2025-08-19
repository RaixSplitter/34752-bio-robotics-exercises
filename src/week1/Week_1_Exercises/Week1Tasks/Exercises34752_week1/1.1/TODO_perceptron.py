import numpy as np
from activation import ActivationFunction
import matplotlib.pyplot as plt


##########################################
"""
TODO:
For each activation function: Sign, Sigmoid, Linear, define the following:
1) Forward function
2) Gradient function

We have provided the function signature for you.

"""
##########################################


class SignActivation(ActivationFunction):
    """
    Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
    """

    def forward(self, x):
        return 1 if x > 0 else 0

    def gradient(self, x):
        """
        Function derivative.
        Define the correct return value (derivative), given input `x`
        """
        return np.zeros_like(
            x
        )  # The gradient of the sign function is zero almost everywhere except at x=0, which is not defined


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.forward(x) * (
            1 - self.forward(x)
        )  # Sigmoid derivative: f'(x) = f(x)(1 - f(x))


class LinearActivation(ActivationFunction):
    def forward(self, x):
        return x

    def gradient(self, x):
        return 1


class Perceptron:
    """
    Perceptron neuron model
    Parameters
    ----------
    n_inputs : int
        Number of inputs
    act_f : Subclass of `ActivationFunction`
        Activation function
    """

    def __init__(self, n_inputs, act_f):
        """
        Perceptron class initialization
        TODO: Write the code to initialize weights and save the given activation function
        """
        if not isinstance(act_f, type) or not issubclass(act_f, ActivationFunction):
            raise TypeError(
                "act_f has to be a subclass of ActivationFunction (not a class instance)."
            )
        # weights: include bias weight w0, so n_inputs + 1
        self.w = np.random.normal(0, 1, size=n_inputs + 1)
        # activation function
        self.f = act_f()  # instantiate the activation function object

        if self.f is not None and not isinstance(self.f, ActivationFunction):
            raise TypeError("self.f should be a class instance.")

    def activation(self, x):
        """
        It computes the activation `a` given an input `x`
        TODO: Fill in the function to provide the correct output
        NB: Remember the bias
        """
        x_aug = np.concatenate(([1], x))  # prepend x0=1 for bias
        a = np.dot(self.w, x_aug)
        return a

    def output(self, a):
        """
        It computes the neuron output `y`, given the activation `a`
        TODO: Fill in the function to provide the correct output
        """
        y = self.f.forward(a)
        return y

    def predict(self, x):
        """
        It computes the neuron output `y`, given the input `x`
        TODO: Fill in the function to provide the correct output
        """
        return self.output(self.activation(x))

    def gradient(self, a):
        """
        It computes the gradient of the activation function, given the activation `a`
        """
        return self.f.gradient(a)


if __name__ == "__main__":
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

    ## TODO Test your activation function
    a = Sigmoid()
    print(a.forward(2))
    "print(a.forward(0))"

    ## TODO Test perceptron initialization
    p = Perceptron(n_inputs=2, act_f=SignActivation)
    print(f"{p.predict(xdata[0, :])}")
    for i in range(len(xdata)):
        print(f"{xdata[i, :]} -> {p.predict(xdata[i, :])}")
    ## TODO Learn the weights
    r = 0.001  # learning rate
    ## calculate the error and update the weights
    print(p.w)
    training_error = []
    for i in range(1000):
        for j in range(len(xdata)):
            x = xdata[j, :]
            y = ydata[j]
            a = p.activation(x)
            y_pred = p.predict(x)
            error = y - y_pred
            x_aug = np.concatenate(([1], x))
            p.w += r * error * x_aug  # Update weights

            training_error.append(float(error**2))

    ## TODO plot points and linear decision boundary
    print(f"{p.predict(xdata[0, :])}")
    for i in range(len(xdata)):
        print(f"{xdata[i, :]} -> {p.predict(xdata[i, :])}")

    xp = np.linspace(-1, 3, 100)
    yp = -(p.w[1] * xp + p.w[0]) / p.w[2]  # Calculate y values for the decision boundary
    plt.scatter(xdata[:, 0], xdata[:, 1], c=ydata, cmap="bwr", edgecolors="k")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Perceptron Decision Boundary")
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.grid()
    plt.plot(xp, yp, "k--")
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.show()
    plt.show()
