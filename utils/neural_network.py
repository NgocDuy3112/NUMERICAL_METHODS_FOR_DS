import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    def __init__(self, func_type):
        self.type = func_type

    def compute(self, y_hat, y):
        if self.type == "binary_cross_entropy":
            m = y.shape[0]
            epsilon = 1e-5
            return -1 / m * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
        elif self.type == "mean_squared_error":
            return 1 / 2 * np.mean((y_hat - y) ** 2)
        elif self.type == "mean_absolute_error":
            return np.mean(np.abs(y_hat - y))
        else:
            raise Exception("Invalid loss function. Valid functions are: binary_cross_entropy, mean_squared_error, mean_absolute_error.")
        
    def gradient(self, y_hat, y):
        if self.type == "binary_cross_entropy":
            m = y.shape[0]
            epsilon = 1e-5
            return -1 / m * (y / (y_hat + epsilon) - (1 - y) / (1 - y_hat + epsilon))
        elif self.type == "mean_squared_error":
            return y_hat - y
        elif self.type == "mean_absolute_error":
            return np.where(y_hat > y, 1, -1)
        else:
            raise Exception("Invalid loss function. Valid functions are: binary_cross_entropy, mean_squared_error, mean_absolute_error.")

class ActivationFunction:
    def __init__(self, func_type):
        self.type = func_type

    def compute(self, x):
        if self.type == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.type == "relu":
            return np.maximum(0, x)
        elif self.type == "tanh":
            return np.tanh(x)
        elif self.type == "softmax":
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        elif self.type == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        else:
            raise Exception("Invalid activation function. Valid functions are: sigmoid, relu, tanh, softmax, leaky_relu.")
    
    def gradient(self, x):
        if self.type == "sigmoid":
            return self.compute(x) * (1 - self.compute(x))
        elif self.type == "relu":
            return np.where(x > 0, 1, 0)
        elif self.type == "tanh":
            return 1 - np.power(self.compute(x), 2)
        elif self.type == "softmax":
            return self.compute(x) * (1 - self.compute(x))
        elif self.type == "leaky_relu":
            return np.where(x > 0, 1, 0.01)
        else:
            raise Exception("Invalid activation function. Valid functions are: sigmoid, relu, tanh, softmax, leaky_relu.")
    

class FullyConnectedNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_layer, loss_type="binary_cross_entropy"):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.num_layers = len(hidden_layers) + 1
        self.loss_func = LossFunction(loss_type)

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        self.activators = []

        # Initialize weights and biases for the input to the first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_layers[0][0]))
        self.biases.append(np.random.randn(hidden_layers[0][0]))
        self.activators.append(ActivationFunction(hidden_layers[0][1]))

        # Initialize weights and biases for the remaining hidden layers
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i][0], hidden_layers[i+1][0]))
            self.biases.append(np.random.randn(hidden_layers[i+1][0]))
            self.activators.append(ActivationFunction(hidden_layers[i+1][1]))

        # Initialize weights and biases for the last hidden layer to the output layer
        self.weights.append(np.random.randn(hidden_layers[-1][0], output_layer[0]))
        self.biases.append(np.random.randn(output_layer[0]))
        self.activators.append(ActivationFunction(output_layer[1]))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        self.grad_values = []

        # Perform forward propagation through each layer
        for i in range(self.num_layers):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activators[i].compute(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        self.deltas = [np.dot(self.loss(y), self.grad_loss(y))]

        # Perform backward propagation through each layer
        for i in range(self.num_layers - 1, 0, -1):
            delta = np.dot(self.deltas[-1], self.weights[i].T) * self.activators[i].gradient(self.activations[i])
            self.deltas.append(delta)

        # Reverse the list of deltas
        self.deltas = self.deltas[::-1]

        self.grad_weights = []
        self.grad_biases = []

        # Compute gradients for weights and biases
        for i in range(self.num_layers):
            grad_w = np.dot(self.activations[i].T, self.deltas[i]) / m
            grad_b = np.mean(self.deltas[i], axis=0)
            self.grad_weights.append(grad_w)
            self.grad_biases.append(grad_b)

    def train(self, X, y, num_epochs, learning_rate, batch_size=None, verbose=False):
        if batch_size is None:
            batch_size = X.shape[0]

        num_batches = X.shape[0] // batch_size
        self.time = np.arange(num_epochs)
        self.losses = []
        self.accuracies = []

        for epoch in range(1, num_epochs + 1):
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                # Update weights and biases for each layer
                for i in range(self.num_layers):
                    self.weights[i] -= learning_rate * self.grad_weights[i]
                    self.biases[i] -= learning_rate * self.grad_biases[i]

            # Print loss after every epoch
            self.forward(X)
            loss = self.loss(y)
            acc = self.accuracy(y)
            self.losses.append(loss)
            self.accuracies.append(acc)
            if verbose:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss}, Accuracy: {acc}")
            
    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.time, self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(self.time, self.accuracies)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

    def predict(self, X):
        # Make predictions
        return np.round(self.forward(X))

    def loss(self, y):
        # Compute loss
        return self.loss_func.compute(self.activations[-1], y)
    
    def grad_loss(self, y):
        # Compute gradient of loss
        return self.loss_func.gradient(self.activations[-1], y)

    def accuracy(self, y):
        # Compute accuracy
        return np.mean(np.round(self.activations[-1]) == y)
    
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    # Create a neural network
    nn = FullyConnectedNeuralNetwork(2, [(4, "leaky_relu"), (2, "tanh")], (1, "sigmoid"))

    # Train the neural network
    nn.train(X, y, num_epochs=500000, learning_rate=0.1, batch_size=2)
    nn.plot_history()

    # Make predictions
    predictions = nn.predict(X)
    print(predictions)
