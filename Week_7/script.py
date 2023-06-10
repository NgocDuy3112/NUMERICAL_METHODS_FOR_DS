import numpy as np

class XORNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        # Initialize weights and biases for the input to the first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros(hidden_sizes[0]))

        # Initialize weights and biases for the remaining hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]))
            self.biases.append(np.zeros(hidden_sizes[i+1]))

        # Initialize weights and biases for the last hidden layer to the output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros(output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        # Perform forward propagation through each layer
        for i in range(self.num_layers):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        self.deltas = [self.activations[-1] - y]

        # Perform backward propagation through each layer
        for i in range(self.num_layers - 1, 0, -1):
            delta = np.dot(self.deltas[-1], self.weights[i].T) * self.activations[i] * (1 - self.activations[i])
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

    def train(self, X, y, num_epochs, learning_rate, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]

        num_batches = X.shape[0] // batch_size

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
            loss = self.cross_entropy_loss(y)
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss}")

    def predict(self, X):
        # Make predictions
        return np.round(self.forward(X))

    def cross_entropy_loss(self, y):
        m = y.shape[0]
        epsilon = 1e-8

        loss = -(np.sum(y * np.log(self.activations[-1] + epsilon) + (1 - y) * np.log(1 - self.activations[-1] + epsilon))) / m
        return loss


if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    # Create a neural network
    nn = XORNeuralNetwork(2, [2], 1)

    # Train the neural network
    nn.train(X, y, num_epochs=10000, learning_rate=0.1)

    # Make predictions
    predictions = nn.predict(X)
    print(predictions)