import numpy as np
from scipy.linalg import svd

def truncated_svd_decomposition(A, information_retained=0.9):
    U, S_vector, Vt = svd(A)
    total_energy = np.sum(S_vector**2)
    energy_retained = 0
    n = 0
    while energy_retained < information_retained * total_energy:
        energy_retained += S_vector[n]**2
        n += 1
    U = U[:, :n]
    S = np.diag(S_vector[:n])
    Vt = Vt[:n, :]
    return U, S, Vt

def create_a_random_matrix(lower_bound, upper_bound, n_rows, n_cols):
    return (upper_bound - lower_bound) * np.random.rand(n_rows, n_cols) + lower_bound

def input_a_matrix():
    n_rows = int(input("Enter the number of rows: "))
    n_cols = int(input("Enter the number of columns: "))
    A = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            A[i, j] = float(input("Enter the element A[{}, {}]: ".format(i, j)))
    return A

if __name__ == "__main__":
    A = np.array([[1, 0, 0, 0, 2],
                  [0, 0, 3, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 2, 0, 0, 0]])
    Uk, Sk, Vtk = truncated_svd_decomposition(A, 0.99)
    print("U = \n", Uk)
    print("S = \n", Sk)
    print("Vt = \n", Vtk)
    print("A = \n", Uk @ Sk @ Vtk)
    print("A = \n", A)