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
        print(X.shape)
        X_new = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        return X_new @ self.w
    

class GradientDescentLinearRegression:
    def __init__(self, algo_type='normal', learning_rate=1e-4, precision=1e-9, epochs=50000, early_stopping=5):
        self.w = None
        self.b = None
        self.algo_type = algo_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.precision = precision
        self.early_stopping = early_stopping
        self.count_iter = 0
    
    def fit(self, X, y, init_point):
        self.w = init_point
        self.b = 0
        best_loss = np.inf
        early_stopping = self.early_stopping

        if self.algo_type == 'normal':            
            for _ in range(self.epochs):
                y_hat = self.predict(X)

                loss = 1 / 2 * np.mean((y_hat - y) ** 2)

                dw = (y_hat - y).T @ X / len(y)
                db = np.sum(y_hat - y) / len(y)

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                if loss < best_loss:
                    best_loss = loss
                    early_stopping = self.early_stopping
                else:
                    early_stopping -= 1
                
                self.count_iter += 1
                
                if early_stopping == 0: break
            
        
        elif self.algo_type == "stochastic":
            for _ in range(self.epochs):
                X_rand = X.iloc[np.random.randint(0, X.shape[0])]
                y_rand = y[np.random.randint(0, y.shape[0])]

                y_hat = self.predict(X_rand)

                loss = 1 / 2 * (y_hat - y_rand) ** 2

                dw = (y_hat - y_rand) * X_rand
                db = (y_hat - y_rand) 

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                if loss < best_loss:
                    best_loss = loss
                    early_stopping = self.early_stopping
                else:
                    early_stopping -= 1
                
                self.count_iter += 1
                
                if early_stopping == 0: break

        elif self.algo_type == "accelerated":
            for epoch in range(self.epochs):
                y_hat = self.predict(X)

                loss = 1 / 2 * np.mean((y_hat - y) ** 2)

                dw = (y_hat - y).T @ X / len(y)
                db = np.sum(y_hat - y) / len(y)

                self.w -= self.learning_rate * dw * (epoch - 1) / (epoch + 2)
                self.b -= self.learning_rate * db * (epoch - 1) / (epoch + 2)

                if loss < best_loss:
                    best_loss = loss
                    early_stopping = self.early_stopping
                else:
                    early_stopping -= 1
                
                self.count_iter += 1
                
                if early_stopping == 0: break
        
        else:
            raise Exception("The algo type is not implemented correcly. Accepted values are normal, stochastic and accelerated")
        print(self.count_iter)

        return self


    def predict(self, X):
        return X @ self.w + self.b


class GradientDescentLogisticRegression:
    def __init__(self, algo_type='normal', learning_rate=1e-4, epochs=50000, early_stopping=5):
        self.w = None
        self.b = None
        self.algo_type = algo_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.count_iter = 0
    
    def fit(self, X, y, init_point):
        self.w = init_point
        self.b = 0
        best_loss = np.inf
        early_stopping = self.early_stopping

        if self.algo_type == 'normal':            
            for _ in range(self.epochs):
                y_hat = self.predict(X)

                loss = - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

                dw = (y_hat - y).T @ X / len(y)
                db = np.sum(y_hat - y) / len(y)

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                if loss < best_loss:
                    best_loss = loss
                    early_stopping = self.early_stopping
                else:
                    early_stopping -= 1
                
                self.count_iter += 1
                
                if early_stopping == 0: break
            
        
        elif self.algo_type == "stochastic":
            for _ in range(self.epochs):
                X_rand = X.iloc[np.random.randint(0, X.shape[0])]
                y_rand = y[np.random.randint(0, y.shape[0])]

                y_hat = self.predict(X_rand)

                loss = - y_rand * np.log(y_hat) - (1 - y_rand) * np.log(1 - y_hat)

                dw = (y_hat - y_rand) * X_rand
                db = (y_hat - y_rand) 

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

                if loss < best_loss:
                    best_loss = loss
                    early_stopping = self.early_stopping
                else:
                    early_stopping -= 1
                
                self.count_iter += 1
                
                if early_stopping == 0: break

        elif self.algo_type == "accelerated":
            for epoch in range(self.epochs):
                y_hat = self.predict(X)

                loss = - y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)

                dw = (y_hat - y).T @ X / len(y)
                db = np.sum(y_hat - y) / len(y)

                self.w -= self.learning_rate * dw * (epoch - 1) / (epoch + 2)
                self.b -= self.learning_rate * db * (epoch - 1) / (epoch + 2)

                if loss < best_loss:
                    best_loss = loss
                    early_stopping = self.early_stopping
                else:
                    early_stopping -= 1
                
                self.count_iter += 1
                
                if early_stopping == 0: break
        
        else:
            raise Exception("The algo type is not implemented correcly. Accepted values are normal, stochastic and accelerated")
        print(self.count_iter)

        return self


    def predict(self, X):
        return 1 / (1 + np.exp(-(X @ self.w + self.b)))