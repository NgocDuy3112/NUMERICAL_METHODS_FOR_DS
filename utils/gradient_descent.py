import numpy as np

class SVDLinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        print(X.shape)
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        U, S_vec, Vt = np.linalg.svd(X_new.T @ X_new)
        S_plus = np.reciprocal(S_vec)
        A_At_plus = Vt.T @ np.diag(S_plus) @ U.T
        self.w = A_At_plus @ X_new.T @ y
        return self
    
    def predict(self, X):
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        return X_new @ self.w
    

class GradientDescentLinearRegression:
    def __init__(self, algo_type='normal', learning_rate=1e-4, precision=1e-9, epochs=50000):
        self.algo_type = algo_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.precision = precision
    
    def fit(self, X, y, init_point):
        self.w = init_point
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        if self.algo_type == 'normal':            
            for _ in range(self.epochs):
                y_hat = X_new @ self.w

                loss = 1 / 2 * np.mean((y_hat - y) ** 2)

                dw = (y_hat - y).T @ X_new / len(y)

                self.w -= self.learning_rate * dw

                if loss < self.precision:
                    break
                    
            
        elif self.algo_type == "stochastic":
            for _ in range(self.epochs):
                X_rand = X_new[np.random.randint(0, X_new.shape[0])]
                y_rand = y[np.random.randint(0, y.shape[0])]

                y_hat = X_rand @ self.w

                loss = 1 / 2 * (y_hat - y_rand) ** 2

                dw = (y_hat - y_rand) * X_rand

                self.w -= self.learning_rate * dw

                if loss < self.precision:
                    break

        elif self.algo_type == "accelerated":
            for epoch in range(self.epochs):
                y_hat = X_new @ self.w

                loss = 1 / 2 * np.mean((y_hat - y) ** 2)

                dw = (y_hat - y).T @ X_new / len(y)

                self.w -= self.learning_rate * dw * (epoch - 1) / (epoch + 2)

                if loss < self.precision:
                    break
        
        else:
            raise Exception("The algo type is not implemented correcly. Accepted values are normal, stochastic and accelerated")
        return self


    def predict(self, X):
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        return X_new @ self.w


class GradientDescentLogisticRegression:
    def __init__(self, algo_type='normal', learning_rate=1e-4, precision=1e-9, epochs=50000):
        self.algo_type = algo_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.precision = precision
    
    def fit(self, X, y, init_point):
        self.w = init_point
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        if self.algo_type == 'normal':            
            for _ in range(self.epochs):
                y_hat = 1 / (1 + np.exp(-(X_new @ self.w)))

                loss = - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

                dw = (y_hat - y).T @ X_new / len(y)
                self.w -= self.learning_rate * dw

                if loss < self.precision:
                    break
            
        elif self.algo_type == "stochastic":
            for _ in range(self.epochs):
                X_rand = X_new[np.random.randint(0, X_new.shape[0])]
                y_rand = y[np.random.randint(0, y.shape[0])]

                y_hat = 1 / (1 + np.exp(-(X_new @ self.w)))

                loss = - y_rand * np.log(y_hat) - (1 - y_rand) * np.log(1 - y_hat)

                dw = (y_hat - y_rand) * X_rand
                self.w -= self.learning_rate * dw

                if loss < self.precision:
                    break

        elif self.algo_type == "accelerated":
            for epoch in range(self.epochs):
                y_hat = 1 / (1 + np.exp(-(X_new @ self.w)))

                loss = - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

                dw = (y_hat - y).T @ X_new / len(y)

                self.w -= self.learning_rate * dw * (epoch - 1) / (epoch + 2)
                
                if loss < self.precision:
                    break
        else:
            raise Exception("The algo type is not implemented correcly. Accepted values are normal, stochastic and accelerated")
        
        return self


    def predict(self, X):
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        return 1 / (1 + np.exp(-(X_new @ self.w)))