import numpy as np
from scipy.linalg import svd

def least_square_solver(A, b):
    U, S_vec, Vt = svd(A.T @ A)
    print(Vt)
    S_plus_vec = np.reciprocal(S_vec)
    A_At_plus = Vt.T @ np.diag(S_plus_vec) @ U.T
    
    y = A_At_plus @ A.T @ b
    return y

if __name__ == "__main__":
    RAM = np.array([1, 3, 4, 3, 3, 6,
                    4, 6, 8, 8, 8, 8,
                    8, 8, 8, 12, 12, 8,
                    8, 8, 12, 12, 12, 12])
    MEMORY = np.array([16, 32, 64, 32, 32, 128,
                       64, 128, 128, 256, 128, 128,
                       256, 128, 128, 128, 256, 256,
                       256, 256, 256, 256, 512, 512])
    PIN = np.array([3000, 4000, 5000, 3300, 5000, 5000,
                       4000, 5000, 4500, 5000, 4500, 4100,
                       4500, 4500, 4500, 5000, 4500, 3300,
                       3300, 4800, 4500, 5000, 5000, 4400])
    bias = np.ones((24, 1))
    PRICE = np.array([1850, 2650, 3350, 3790, 4250, 4700,
                      4150, 5150, 6600, 10100, 10500, 12400,
                      13650, 12790, 15500, 16000, 18990, 19350,
                      20990, 23000, 23000, 29800, 29990, 33990])
    X = np.concatenate((RAM.reshape(24, 1), MEMORY.reshape(24, 1), PIN.reshape(24, 1), bias), axis=1)
    y = PRICE.reshape(24, 1)
    w = least_square_solver(X, y)
    X0 = np.array([4, 64, 4000, 1])
    print(f"Price of laptop with RAM = 4GB, Memory = 64GB, PIN = 4000 is: {int(X0 @ w * 1000)} VND")