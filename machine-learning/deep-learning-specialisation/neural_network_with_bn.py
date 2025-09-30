from pathlib import Path
from typing import Callable
import logging

import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import expit
from IPython.display import display


logger = logging.getLogger()


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    return 1.0 * (x > 0)


sigmoid = expit


def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


def leaky_relu(x, leaky_constant: float = 0.01):
    return np.where(x > 0.0, x, x * leaky_constant)


def leaky_relu_derivative(x, leaky_constant=0.01):
    return np.where(x > 0, 1, leaky_constant)


def log_loss(A, Y, Ws, regularisation_lambda: float = 0):
    m = A.shape[1]  # m is number of samples
    # Clip values to avoid log(0)
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)

    # For multi-class (softmax), we use categorical cross-entropy
    if A.shape[0] > 1:  # If output dimension > 1, assume multi-class
        # Categorical cross-entropy: -1/m * sum(y_i * log(a_i))
        cost = -np.sum(Y * np.log(A)) / m
    else:  # Binary classification case
        cost = -(1 / m) * (Y @ np.log(A).T + (1 - Y) @ np.log(1 - A).T)

    # Add L2 regularization if specified
    if regularisation_lambda:
        l2_cost = (regularisation_lambda / (2 * m)) * sum(
            np.square(W).sum() for W in Ws.values()
        )
        cost += l2_cost

    return np.array([[cost]])


def square_loss(A, Y, Ws, regularisation_lambda: float = 0):
    m = A.shape[1]  # m is number samples
    cost = -(1 / m) * ((A - Y) @ (A - Y).T)
    if not regularisation_lambda:
        return cost
    return cost + (regularisation_lambda / (2 * m)) * sum(
        np.square(W).sum() for W in Ws.values()
    )


def softmax(z):
    # if Z has shape (C, m) then we want to return a matrix of shape (C, m)
    # Note the softmax is applied for each column of Z
    stable_z = z - np.max(z, axis=0, keepdims=True)
    exponents = np.exp(stable_z)
    probabilities = exponents / np.sum(exponents, axis=0, keepdims=True)
    return probabilities


ACTIVATION_FUNCTIONS: dict[str, Callable] = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": expit,
    "softmax": softmax,
}

ACTIVATION_FUNCTION_DERIVATIVES: dict[str, Callable] = {
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "sigmoid": sigmoid_derivative,
}

LOSS_FUNCTIONS: dict[str, Callable] = {"log_loss": log_loss, "square_loss": square_loss}


class NeuralNetwork:
    """This implements a simple feedforward neural network.

    This additionally allows for dropout, regularisation, batch normalization,
    and multi class output via the softmax function.
    """

    layer_activations: dict[int, str]

    def __init__(
        self,
        layer_sizes: list[int],
        layer_activations: dict[int, str] | None = None,
        regularisation_lambda: float = 0.0,
        keep_prob: float = 1.0,
        cost_function: str = "log_loss",
        model_id: str = "",
        batch_norm_layers: list[int] | None = None,
        batch_norm_momentum: float = 0.9,
    ):
        self.layer_sizes = layer_sizes
        self.regularisation_lambda = regularisation_lambda
        self.L = len(layer_sizes) - 1
        self.m = layer_sizes[0]
        self.cost_function = cost_function
        self.keep_prob = keep_prob
        self.model_id = model_id
        self.batch_norm_layers = batch_norm_layers or []
        self.batch_norm_momentum = batch_norm_momentum
        self.training = True

        if layer_activations:
            self.layer_activations = layer_activations
        else:
            # This sets all hidden layers and the output layer to "sigmoid" by default.
            self.layer_activations = {l: "sigmoid" for l in range(1, self.L)} | {
                self.L: "sigmoid"
            }

    def initialise_weights(self) -> None:
        # This is using He initialisation.
        # Try changing to * 0.01 and see the change in cost plot.
        self.params = {}
        for l, (n_l, n_l_minus_1) in enumerate(
            zip(self.layer_sizes[1:], self.layer_sizes), start=1
        ):
            self.params[f"W{l}"] = np.random.normal(size=(n_l, n_l_minus_1)) * np.sqrt(
                2 / n_l_minus_1
            )
            self.params[f"b{l}"] = np.zeros((n_l, 1))

        for l in self.batch_norm_layers:
            if l <= self.L:  # Don't apply to output layer typically
                n_l = self.layer_sizes[l]
                self.params[f"gamma{l}"] = np.ones((n_l, 1))
                self.params[f"beta{l}"] = np.zeros((n_l, 1))
                # Running statistics for inference
                self.params[f"running_mean{l}"] = np.zeros((n_l, 1))
                self.params[f"running_var{l}"] = np.ones((n_l, 1))

        logger.info("Weights initialised")
        logger.debug(f"{self.params=}")

    def forward(self, X) -> None:
        Zs, As, Ds, batch_norm_cache = {}, {0: X}, {}, {}

        for l in range(1, self.L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z_raw = W @ As[l - 1] + b

            # NEW: Apply batch normalization if specified for this layer
            if l in self.batch_norm_layers:
                if self.training:
                    # Training mode: compute batch statistics
                    mu = np.mean(Z_raw, axis=1, keepdims=True)
                    var = np.var(Z_raw, axis=1, keepdims=True)

                    # Update running statistics
                    self.params[f"running_mean{l}"] = (
                        self.batch_norm_momentum * self.params[f"running_mean{l}"]
                        + (1 - self.batch_norm_momentum) * mu
                    )
                    self.params[f"running_var{l}"] = (
                        self.batch_norm_momentum * self.params[f"running_var{l}"]
                        + (1 - self.batch_norm_momentum) * var
                    )
                else:
                    # Inference mode: use running statistics
                    mu = self.params[f"running_mean{l}"]
                    var = self.params[f"running_var{l}"]

                # Normalize
                epsilon = 1e-8
                Z_norm = (Z_raw - mu) / np.sqrt(var + epsilon)

                # Scale and shift
                gamma = self.params[f"gamma{l}"]
                beta = self.params[f"beta{l}"]
                Z_tilde = gamma * Z_norm + beta  # Final batch normalized output
                Zs[l] = Z_tilde  # Store Z_tilde for activation

                # Cache for backprop
                if self.training:
                    batch_norm_cache[l] = {
                        "Z_raw": Z_raw,
                        "Z_norm": Z_norm,
                        "mu": mu,
                        "var": var,
                        "epsilon": epsilon,
                    }
            else:
                Zs[l] = Z_raw

            # Apply activation function
            g = ACTIVATION_FUNCTIONS[self.layer_activations[l]]
            logger.debug(f"Applying {self.layer_activations[l]} in layer{l}")
            As[l] = g(Zs[l])

            # apply drop out but not on the output layer
            if self.training and l != self.L:
                Ds[l] = (np.random.uniform(size=As[l].shape) < self.keep_prob).astype(
                    int
                ) / self.keep_prob
                As[l] *= Ds[l]

        if self.training:
            self.Zs, self.As, self.Ds, self.batch_norm_cache = (
                Zs,
                As,
                Ds,
                batch_norm_cache,
            )
        return As[self.L]

    def backward(self, Y):
        dZs = {self.L: self.As[self.L] - Y}
        m = self.As[0].shape[1]
        grads = {}

        for l in range(self.L, 0, -1):
            logger.debug(f"calculating dZ for layer_id {l}")
            if l != self.L:
                W_l_plus_1 = self.params[f"W{l + 1}"]
                dA_l = W_l_plus_1.T @ dZs[l + 1]

                # Apply dropout gradient if used
                if hasattr(self, "Ds") and l in self.Ds:
                    dA_l *= self.Ds[l]

                # Activation derivative (gradient w.r.t. Z_tilde)
                dZ_tilde = dA_l * ACTIVATION_FUNCTION_DERIVATIVES[
                    self.layer_activations[l]
                ](
                    self.Zs[l]
                )  # Note: Zs[l] stores Z_tilde

                # NEW: Batch normalization backward pass
                if l in self.batch_norm_layers:
                    # Get cached values
                    cache = self.batch_norm_cache[l]
                    Z_raw = cache["Z_raw"]
                    Z_norm = cache["Z_norm"]
                    mu = cache["mu"]
                    var = cache["var"]
                    epsilon = cache["epsilon"]

                    gamma = self.params[f"gamma{l}"]

                    # Gradients for gamma and beta
                    grads[f"dgamma{l}"] = (
                        np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True) / m
                    )
                    grads[f"dbeta{l}"] = np.sum(dZ_tilde, axis=1, keepdims=True) / m

                    # Gradient for the normalized input
                    dZ_norm = dZ_tilde * gamma

                    # Gradient for the input before normalization (this is the tricky part)
                    dvar = np.sum(
                        dZ_norm * (Z_raw - mu) * (-0.5) * (var + epsilon) ** (-1.5),
                        axis=1,
                        keepdims=True,
                    )
                    dmu = (
                        np.sum(
                            dZ_norm * (-1.0 / np.sqrt(var + epsilon)),
                            axis=1,
                            keepdims=True,
                        )
                        + dvar * np.sum(-2.0 * (Z_raw - mu), axis=1, keepdims=True) / m
                    )

                    dZs[l] = (
                        dZ_norm / np.sqrt(var + epsilon)
                        + dvar * 2.0 * (Z_raw - mu) / m
                        + dmu / m
                    )
                else:
                    dZs[l] = dZ_tilde  # No batch norm, so dZ = dZ_tilde

            # Standard gradients for weights and biases
            grads[f"dW{l}"] = (1.0 / m) * dZs[l] @ self.As[l - 1].T
            if self.regularisation_lambda and l != self.L:
                grads[f"dW{l}"] += (self.regularisation_lambda / m) * self.params[
                    f"W{l}"
                ]
            grads[f"db{l}"] = (1.0 / m) * np.sum(dZs[l], axis=1, keepdims=True)

        return grads

    # NEW: Training mode control methods
    def train_mode(self):
        """Set model to training mode."""
        self.training = True

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.training = False

    def cost(self, A, Y) -> float:
        # For cost, we need to pass the weights. We'll extract them from self.params.
        Ws = {l: self.params[f"W{l}"] for l in range(1, self.L + 1)}
        if self.cost_function == "log_loss":
            return log_loss(A, Y, Ws, self.regularisation_lambda).item()
        elif self.cost_function == "square_loss":
            return square_loss(A, Y, Ws, self.regularisation_lambda).item()
        else:
            raise Exception(
                f"Incorrect value for self.cost_function:= {self.cost_function}"
            )

    def predict(self, X, return_probability=False):
        # Set to eval mode for prediction
        original_mode = self.training
        self.eval_mode()
        try:
            Y_hat = self.forward(X)
            if return_probability:
                return Y_hat
            return np.where(Y_hat > 0.5, 1, 0)
        finally:
            # Restore original mode
            self.training = original_mode


class Optimizer:
    """This implements gradient descent. With optional mini-batching."""

    def __init__(self, learning_rate: float = 0.1, batch_size: int | None = None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def update_model_params(self, model, grads, training_iteration):
        """Update model parameters using grads returned from backward."""
        for param_key in model.params:
            grad_key = f"d{param_key}"
            if (
                grad_key in grads
            ):  # NEW: check if gradient exists (handles batch norm params)
                model.params[param_key] -= self.learning_rate * grads[grad_key]

    def train(
        self,
        model,
        X,
        Y,
        n_epochs=1000,
        log_every: int | None = None,
        plot_cost=False,
        fig=None,
        plot_every=10,
        shuffle=True,
    ):
        costs, epochs = [], []
        model.train_mode()
        if plot_cost:
            if fig is None:
                fig = go.FigureWidget()
                display(fig)
            fig.add_scatter(x=[], y=[], mode="lines+markers", name=model.model_id)
            fig.update_layout(
                title="Training Cost over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Cost",
            )

        m = X.shape[1]
        batch_size = self.batch_size if self.batch_size is not None else m

        training_iteration = 1
        for epoch in range(n_epochs):
            if shuffle:
                indices = np.random.permutation(m)
                X = X[:, indices]
                Y = Y[:, indices]
            for i in range(0, m, batch_size):
                X_batch = X[:, i : i + batch_size]
                Y_batch = Y[:, i : i + batch_size]

                # Forward pass
                A = model.forward(X_batch)
                grads = model.backward(Y_batch)
                self.update_model_params(model, grads, training_iteration)
                training_iteration += 1

            # Compute cost on the whole dataset after epoch
            model.eval_mode()
            A_full = model.forward(X)
            cost = model.cost(A_full, Y)
            model.train_mode()
            if log_every and epoch % log_every == 0:
                logger.info(f"Cost after epoch {epoch} = {cost}")
            if plot_cost and fig is not None:
                costs.append(cost)
                epochs.append(epoch + 1)
                if epoch % plot_every == 0:
                    with fig.batch_update():
                        fig.data[-1].x = epochs  # type: ignore
                        fig.data[-1].y = costs  # type: ignore
        model.eval_mode()


class MomentumOptimizer(Optimizer):
    """Implements momentum optimizer.

    The update rule is:
    v_t = beta * v_{t-1} + grad_t
    param_t = param_{t-1} - learning_rate * v_t
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        batch_size: int | None = None,
        beta: float = 0.9,
    ):
        super().__init__(learning_rate, batch_size)
        self.beta = beta
        self.cache = {}

    def update_model_params(self, model, grads, training_iteration):
        for param_key in model.params:
            grad_key = f"d{param_key}"
            if grad_key in grads:  # NEW: check if gradient exists
                if param_key not in self.cache:
                    self.cache[param_key] = np.zeros_like(model.params[param_key])
                self.cache[param_key] = (
                    self.beta * self.cache[param_key]
                    + (1 - self.beta) * grads[grad_key]
                )
                model.params[param_key] -= self.learning_rate * self.cache[param_key]


class RMSPropOptimizer(Optimizer):
    """Implements RMSProp optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        batch_size: int | None = None,
        beta: float = 0.9,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate, batch_size)
        self.beta = beta
        self.epsilon = epsilon  # to avoid division by zero
        self.s_cache = {}

    def update_model_params(self, model, grads, training_iteration):
        for param_key in model.params:
            grad_key = f"d{param_key}"
            if grad_key in grads:  # NEW: check if gradient exists
                if param_key not in self.s_cache:
                    self.s_cache[param_key] = np.zeros_like(model.params[param_key])
                self.s_cache[param_key] = self.beta * self.s_cache[param_key] + (
                    1 - self.beta
                ) * (grads[grad_key] ** 2)
                model.params[param_key] -= (
                    self.learning_rate
                    * grads[grad_key]
                    / (np.sqrt(self.s_cache[param_key]) + self.epsilon)
                )


class ADAMOptimizer(Optimizer):
    """Implements ADAM optimizer with learning rate decay."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        batch_size: int | None = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        learning_rate_decay: float = 0,
    ):
        super().__init__(learning_rate, batch_size)
        self.beta_1 = beta_1  # for the momentum
        self.beta_2 = beta_2  # for the second moment (RMSProp)
        self.epsilon = epsilon  # to avoid division by zero
        self.learning_rate_decay = learning_rate_decay
        self.v_cache = {}
        self.s_cache = {}

    def update_model_params(self, model, grads, training_iteration):
        learning_rate = self.learning_rate * (
            1 / (1 + self.learning_rate_decay * training_iteration)
        )
        for param_key in model.params:
            grad_key = f"d{param_key}"
            if grad_key in grads:  # NEW: check if gradient exists
                if param_key not in self.v_cache:
                    self.v_cache[param_key] = np.zeros_like(model.params[param_key])
                    self.s_cache[param_key] = np.zeros_like(model.params[param_key])
                self.v_cache[param_key] = (
                    self.beta_1 * self.v_cache[param_key]
                    + (1 - self.beta_1) * grads[grad_key]
                )
                self.s_cache[param_key] = (
                    self.beta_2 * self.s_cache[param_key]
                    + (1 - self.beta_2) * grads[grad_key] ** 2
                )
                v_t_corrected = self.v_cache[param_key] / (
                    1 - self.beta_1**training_iteration
                )
                s_t_corrected = self.s_cache[param_key] / (
                    1 - self.beta_2**training_iteration
                )

                model.params[param_key] -= (
                    learning_rate
                    * v_t_corrected
                    / (np.sqrt(s_t_corrected) + self.epsilon)
                )


def plot_decision_boundary(model, X, y, name=None, resolution=0.02):
    """
    Plot the decision boundary for a model trained on all possible pairs of features.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import itertools

    n_features = X.shape[0]
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    class_names = ["setosa", "versicolor", "virginica"]
    colors = ["red", "blue", "green"]

    feature_pairs = list(itertools.combinations(range(n_features), 2))
    n_pairs = len(feature_pairs)
    n_rows = (n_pairs + 1) // 2
    n_cols = 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[
            f"{feature_names[i]} vs {feature_names[j]}" for i, j in feature_pairs
        ],
    )
    fig.update_layout(
        title=(
            f"Decision Boundaries for {name}"
            if name
            else "Decision Boundaries for All Feature Pairs"
        )
    )

    y_classes = np.argmax(y, axis=0)
    mean_features = X.mean(axis=1)

    for idx, (feat1, feat2) in enumerate(feature_pairs):
        row = idx // 2 + 1
        col = idx % 2 + 1

        X_plot = X[[feat1, feat2], :]

        # Create mesh grid
        x_min, x_max = X_plot[0].min() - 0.5, X_plot[0].max() + 0.5
        y_min, y_max = X_plot[1].min() - 0.5, X_plot[1].max() + 0.5

        # Create mesh grid points
        x_mesh = np.linspace(x_min, x_max, 100)
        y_mesh = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_mesh, y_mesh)

        # Prepare input for prediction
        mesh_points = np.vstack([xx.ravel(), yy.ravel()])
        X_pred = np.tile(mean_features.reshape(-1, 1), (1, mesh_points.shape[1]))
        X_pred[feat1] = mesh_points[0]
        X_pred[feat2] = mesh_points[1]

        # Get predictions
        Z_probs = model.predict(X_pred, return_probability=True)
        Z = np.argmax(Z_probs, axis=0)
        Z = Z.reshape(100, 100)

        # Plot decision boundary
        contour = fig.add_trace(
            go.Contour(
                x=x_mesh,
                y=y_mesh,
                z=Z,
                colorscale=[
                    [0, "rgb(255,0,0)"],  # red for setosa
                    [0.5, "rgb(0,0,255)"],  # blue for versicolor
                    [1, "rgb(0,255,0)"],  # green for virginica
                ],
                opacity=0.4,
                colorbar=(
                    dict(
                        title="Predicted Class",
                        ticktext=class_names,
                        tickvals=[0, 1, 2],
                        tickmode="array",
                        len=1.0,
                        yanchor="top",
                        y=1,
                    )
                    if idx == 0
                    else None
                ),
                contours=dict(
                    showlabels=True,
                ),
            ),
            row=row,
            col=col,
        )

        # Plot training points
        for i in range(3):
            mask = y_classes == i
            fig.add_trace(
                go.Scatter(
                    x=X_plot[0, mask],
                    y=X_plot[1, mask],
                    mode="markers",
                    name=class_names[i],
                    marker=dict(
                        size=8,
                        color=colors[i],
                    ),
                    showlegend=True if idx == 0 else False,
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title="Decision Boundaries for All Feature Pairs",
        showlegend=True,
        legend=dict(
            title="Actual Classes", yanchor="top", y=0.99, xanchor="left", x=1.05
        ),
    )

    # Update axes labels
    for idx, (feat1, feat2) in enumerate(feature_pairs):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.update_xaxes(title_text=feature_names[feat1], row=row, col=col)
        fig.update_yaxes(title_text=feature_names[feat2], row=row, col=col)

    fig.show()


def softmax_regression_example():
    """Original softmax regression example without batch normalization."""
    data = sns.load_dataset("iris")
    np.random.seed(45)
    # one hot encode the y values
    data = pd.get_dummies(data, columns=["species"]).astype(float)
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)
    X_train = train_data[["sepal_length", "sepal_width"]].to_numpy().T
    y_train = (
        train_data[["species_setosa", "species_versicolor", "species_virginica"]]
        .to_numpy()
        .T
    )

    X_test = test_data[["sepal_length", "sepal_width"]].to_numpy().T
    y_test = (
        test_data[["species_setosa", "species_versicolor", "species_virginica"]]
        .to_numpy()
        .T
    )
    print(f"X_train.shape: {X_train.shape}")

    layers = [2, 15, 5, 3]
    nn = NeuralNetwork(
        layers,
        model_id="Softmax regression",
        layer_activations={1: "relu", 2: "relu", 3: "softmax"},
    )
    nn.initialise_weights()
    nn_adam_optimizer = ADAMOptimizer(batch_size=60, learning_rate=0.001)

    nn_batch_norm = NeuralNetwork(
        layers,
        model_id="Softmax regression with batch normalization",
        layer_activations={1: "relu", 2: "relu", 3: "softmax"},
        batch_norm_layers=[1, 2],
        batch_norm_momentum=0.9,
    )
    nn_batch_norm.initialise_weights()
    nn_batch_norm_adam_optimizer = ADAMOptimizer(batch_size=60, learning_rate=0.001)

    fig = go.FigureWidget()

    from time import perf_counter

    start_time = perf_counter()
    nn_adam_optimizer.train(
        nn, X_train, y_train, n_epochs=2000, plot_cost=True, fig=fig
    )
    end_time = perf_counter()
    time = end_time - start_time

    start_time = perf_counter()
    nn_batch_norm_adam_optimizer.train(
        nn_batch_norm, X_train, y_train, n_epochs=2000, plot_cost=True, fig=fig
    )
    end_time = perf_counter()
    time_norm = end_time - start_time
    fig.show()

    print(f"Training time: {time:.4f} seconds")
    print(f"Training time with batch normalization: {time_norm:.4f} seconds")

    for name, model in zip(
        ["Original", "With batch normalization"], [nn, nn_batch_norm]
    ):
        # Get predictions
        train_Y_pred = model.predict(X_train)
        test_Y_pred = model.predict(X_test)
        # Convert one-hot encoded targets to class indices
        y_train_classes = np.argmax(y_train, axis=0)
        y_test_classes = np.argmax(y_test, axis=0)

        train_Y_pred_classes = np.argmax(train_Y_pred, axis=0)
        test_Y_pred_classes = np.argmax(test_Y_pred, axis=0)

        print(f"train_Y_pred_classes: {train_Y_pred_classes}")

        # Calculate accuracies using class indices
        train_accuracy = (train_Y_pred_classes == y_train_classes).mean()
        test_accuracy = (test_Y_pred_classes == y_test_classes).mean()

        print(f"Accuracy with {name} on train: {train_accuracy}")
        print(f"Accuracy with {name} on test: {test_accuracy}")

        # plot the decision boundary
        plot_decision_boundary(model, X_train, y_train, name=name)


def regression_example():
    """Original softmax regression example without batch normalization."""
    path = Path(__file__).parent.parent / "titanic" / "processed"
    X_train = pd.read_feather(path / "X_train.feather").to_numpy().T
    y_train = pd.read_feather(path / "y_train.feather").to_numpy().T
    X_test = pd.read_feather(path / "X_test.feather").to_numpy().T
    y_test = pd.read_feather(path / "y_test.feather").to_numpy().T
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_test.shape: {y_test.shape}")

    layers = [30, 50, 20, 1]
    nn = NeuralNetwork(
        layers,
        model_id="Numeral net",
    )
    nn.initialise_weights()
    nn_adam_optimizer = ADAMOptimizer(batch_size=178, learning_rate=0.01)

    nn_batch_norm = NeuralNetwork(
        layers,
        model_id="Numeral net with batch normalization",
        batch_norm_layers=[1, 2],
        batch_norm_momentum=0.5,
    )
    nn_batch_norm.initialise_weights()
    nn_batch_norm_adam_optimizer = ADAMOptimizer(batch_size=178, learning_rate=0.01)

    fig = go.FigureWidget()

    from time import perf_counter

    start_time = perf_counter()
    nn_adam_optimizer.train(
        nn, X_train, y_train, n_epochs=1000, plot_cost=True, fig=fig
    )
    end_time = perf_counter()
    time = end_time - start_time

    start_time = perf_counter()
    nn_batch_norm_adam_optimizer.train(
        nn_batch_norm, X_train, y_train, n_epochs=1000, plot_cost=True, fig=fig
    )
    end_time = perf_counter()
    time_norm = end_time - start_time
    fig.show()

    print(f"Training time: {time:.4f} seconds")
    print(f"Training time with batch normalization: {time_norm:.4f} seconds")

    for name, model in zip(
        ["Original", "With batch normalization"], [nn, nn_batch_norm]
    ):
        # Get predictions
        train_Y_pred = model.predict(X_train)
        test_Y_pred = model.predict(X_test)
        train_accuracy = (train_Y_pred == y_train).sum() / y_train.shape[1]
        test_accuracy = (test_Y_pred == y_test).sum() / y_test.shape[1]
        print(f"train_Y_pred.shape: {train_Y_pred.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"test_Y_pred.shape: {test_Y_pred.shape}")
        print(f"y_test.shape: {y_test.shape}")
        print(f"Accuracy with {name} on train: {train_accuracy}")
        print(f"Accuracy with {name} on test: {test_accuracy}")
        print(f"First 5 predictions: {train_Y_pred[:, :5]}")
        print(f"First 5 targets: {y_train[:, :5]}")


softmax_regression_example()
regression_example()
