import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, func, deriv, learning_rate=0.01, precision=0.0001, max_iters=10000, backtracking=False):
        self.func = func
        self.deriv = deriv
        self.learning_rate = learning_rate
        self.precision = precision
        self.max_iters = max_iters
        self.backtracking = backtracking
        self.iterations = 0
        self.x_process = []
        self.y_process = []

    def __backtracking__(self, x, alpha, beta):
        grad = self.deriv(x)
        norm_grad = np.linalg.norm(grad, 2)
        t = 1
        while self.func(x - t * grad) > self.func(x) - alpha * t * norm_grad ** 2:
            t *= beta
        self.learning_rate = t
        return t        
    
    def optimize(self, iters, x, alpha=0, beta=0):
        x_new = x
        self.x_process.append(x_new)
        self.y_process.append(self.func(x_new))
        if iters > self.max_iters:
            iters = self.max_iters
        for _ in range(iters):
            if self.backtracking:
                self.__backtracking__(alpha, beta, x_new)
            x_new = x_new - self.learning_rate * self.deriv(x_new)
            if np.linalg.norm(self.deriv(x_new), 2) < self.precision:
                break
            self.iterations += 1
            self.x_process.append(x_new)
            self.y_process.append(self.func(x_new))
        return x_new
    
    def get_iterations(self):
        return self.iterations
    
    def get_process(self):
        return self.x_process, self.y_process

# class GradientDescent:
#     def __init__(self, func, deriv, learning_rate=0.01, precision=0.0001, max_iters=10000):
#         self.func = func
#         self.deriv = deriv
#         self.learning_rate = learning_rate
#         self.precision = precision
#         self.max_iters = max_iters
    
#     def optimize(self, iters, x):
#         x_new = x
#         if iters > self.max_iters:
#             iters = self.max_iters
#         for _ in range(iters):
#             x_new = x_new - self.learning_rate * self.deriv(x_new)
#             if np.linalg.norm(self.deriv(x_new), 2) < self.precision:
#                 break
#         return x_new
    
class AcceleratedGradientDescent:
    def __init__(self, func, deriv, learning_rate=0.01, precision=0.0001, max_iters=10000):
        self.func = func
        self.deriv = deriv
        self.learning_rate = learning_rate
        self.precision = precision
        self.max_iters = max_iters
        self.iterations = 0

    def optimize(self, iters, x):
        if iters > self.max_iters:
            iters = self.max_iters
        x_k = x
        y_k = x
        for i in range(iters):
            x_k = y_k - self.learning_rate * self.deriv(y_k)
            y_k = x_k + i / (i + 3) * (x_k - x)
            if np.linalg.norm(self.deriv(x_k), 2) < self.precision:
                break
            self.iterations += 1
        return x_k
    
    def get_iterations(self):
        return self.iterations
    
def func1(x):
    return np.atleast_1d(x ** 2)

def deriv1(x):
    return np.atleast_1d(2 * x)

def func2(x):
    return np.atleast_1d(x ** 4 - 5 * x ** 2 - 3 * x)

def deriv2(x):
    return np.atleast_1d(4 * x ** 3 - 10 * x - 3)

def func3(x):
    return np.atleast_1d(x - np.log(x))

def deriv3(x):
    return np.atleast_1d(1 - 1 / x)

def benchmark(algo, func, deriv, learning_rate, epsilon, max_iters, x, print_output=False, ax=None):
    optimizer = algo(func, deriv, learning_rate, epsilon, max_iters)
    x_opt = optimizer.optimize(max_iters, x)
    n_iters = optimizer.get_iterations()
    if print_output:
        print("Learning rate: ", learning_rate)
        print("Initial point: ", x)
        print("N: ", max_iters)
        print("Epsilon: ", epsilon)
        print("Iterations: ", n_iters)
        print("Solution: ", x_opt)
    if ax is not None:
        tx = np.linspace(-10, 10, 100)
        ty = func(tx)
        ax.plot(tx, ty, 'k-')
        ax.plot(x, func(x), 'go')
        ax.plot(optimizer.get_process()[0], optimizer.get_process()[1], 'bo')
        ax.plot(x_opt, func(x_opt), 'rx', markersize=20, linewidth=5)        
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
    return pd.Series([learning_rate, max_iters, epsilon, x, n_iters, x_opt], 
                     index=["Learning rate", "N", "Epsilon", "Initial_Point", "Iterations", "Solution"])


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    params = [(0.1, 1e-3, 100, -5), (0.01, 1e-3, 100, -5), (0.01, 1e-3, 200, -5),
              (0.1, 1e-3, 100, 5), (0.01, 1e-3, 100, 5), (0.01, 1e-3, 200, 5)]
    results = pd.DataFrame(columns=["Learning rate", "N", "Epsilon", "Initial_Point", "Iterations", "Solution"])
    for i, param in enumerate(params):
        result1 = benchmark(GradientDescent, func1, deriv1, *param, ax=axs[i // 3, i % 3])
        results = results.append(result1, ignore_index=True)
    plt.show()
    best_gd = GradientDescent(func1, deriv1, backtracking=True)
    best_gd.optimize(100, -5)
    print(results)
    print(best_gd.__backtracking__(0.1, 0.5, -5))
