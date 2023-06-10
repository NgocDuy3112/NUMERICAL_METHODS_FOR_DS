import numpy as np
from scipy.linalg import svd, toeplitz
import matplotlib.pyplot as plt
from PIL import Image

# A: decrease matrix
# B: blur image
# X: original image

def decrease_matrix(image, l):
    m = image.shape[1]
    n = m + l - 1
    A = np.zeros((m, n))
    for i in range(m):
        A[i, i:i + l] = 1 / l
    return A

def find_psudo_inverse(A):
    U, S_vec, Vt = svd(A, full_matrices=False)
    S = np.diag(S_vec)
    S_plus = np.zeros(S.shape)
    for i in range(S.shape[0]):
        if S[i, i] != 0:
            S_plus[i, i] = 1 / S[i, i]
    A_plus = Vt.T @ S_plus @ U.T
    return A_plus

def recover_image(gray_image, l, option='h'):
    A = decrease_matrix(gray_image, l)
    A_plus = find_psudo_inverse(A)
    if option == 'h':
        X = gray_image @ A_plus.T
    elif option == 'v':
        X = A_plus @ gray_image
    else:
        raise ValueError('Option must be h or v')
    return X

def grayscale(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

def comparision_plot(blur_image, original_image):
    plt.subplot(1, 2, 1)
    plt.imshow(blur_image, cmap='gray')
    plt.title('Blur image')
    plt.subplot(1, 2, 2)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original image')
    plt.show()

if __name__ == "__main__":
    blur_image = Image.open('parrot_vertical.jpg')
    gray_image = grayscale(np.array(blur_image))
    new_image = recover_image(gray_image, 30, 'v')
    comparision_plot(gray_image, new_image)