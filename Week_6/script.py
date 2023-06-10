import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class GradientDescentLogisticRegression:
    def __init__(self, algo_type='normal', learning_rate=2e-4, epochs=50000, early_stopping=5, threshold=0.5):
        self.w = None
        self.b = None
        self.algo_type = algo_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.count_iter = 0
        self.threshold = threshold
    
    def fit(self, X, y, init_point):
        self.w = init_point
        self.b = 0
        best_loss = np.inf
        early_stopping = self.early_stopping

        if self.algo_type == 'normal':            
            for _ in range(self.epochs):
                y_hat = self.predict(X)

                loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

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

                loss = -np.sum(y_rand * np.log(y_hat) + (1 - y_rand) * np.log(1 - y_hat))

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

                loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

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

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-(X @ self.w + self.b)))
    
    def predict(self, X):
        return self.predict_proba(X) >= self.threshold


if __name__ == "__main__":
    threshold = 0.8
    data = pd.read_csv("datasets/bai_3.csv")
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, -1].values
    
    fig, axs = plt.subplots(1, 2)

    agd_lr = GradientDescentLogisticRegression(threshold=threshold, algo_type = "accelerated")
    agd_lr.fit(X, y, init_point=np.array([-1.0, 5.0]))
    y_pred_agd_lr = agd_lr.predict(X)

    rhs = -np.log(1 / threshold - 1)

    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred_agd_lr)
    plt.xlabel("Thoi gian lam viec")
    plt.ylabel("Muc luong")

    t = np.linspace(2.5, 11, 100)
    yt = (rhs - agd_lr.b - agd_lr.w[0] * t) / agd_lr.w[1]
    plt.plot(t, yt)
    plt.title("Using Accelerated Gradient Descent")
