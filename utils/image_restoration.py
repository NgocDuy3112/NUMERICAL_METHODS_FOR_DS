from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageRestoration:
    def __init__(self, blur_image, l=5, type='horizontal'):
        self.blur_image = blur_image
        self.l = l
        self.A = self.__create_toeplitz_matrix__(l)
        self.type = type

    def __create_toeplitz_matrix__(self, l=30):
        gray_image = 0.299 * self.blur_image[:, :, 0] + 0.587 * self.blur_image[:, :, 1] + 0.114 * self.blur_image[:, :, 2]
        m = gray_image.shape[1]
        n = gray_image.shape[1] + l - 1

        A = np.zeros((m, n))
        for i in range(m):
            for j in range(i, i+l):
                A[i, j] = 1 / l
        return A

    def __restore__(self, X, l=30):
        A = self.__create_toeplitz_matrix__(l)
        U, S_vec, Vt = np.linalg.svd(A, full_matrices=False)
        S_pinv = np.diag(1/S_vec)
        A_pinv = Vt.T @ S_pinv @ U.T
        if self.type == 'horizontal':
            return X @ A_pinv.T
        elif self.type == 'vertical':
            return A_pinv @ X
        else:
            raise ValueError('Type must be horizontal or vertical')
        
    def restore(self):
        R = self.__restore__(self.blur_image[:, :, 0], self.l)
        G = self.__restore__(self.blur_image[:, :, 1], self.l)
        B = self.__restore__(self.blur_image[:, :, 2], self.l)

        restored_img = np.stack([R, G, B], axis=-1)
        restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)
        restored_img = Image.fromarray(restored_img)
        return restored_img
    
    def show(self):
        plt.imshow(self.restore())
        plt.title('Restored Image')
        plt.show()

if __name__ == '__main__':
    img = Image.open("parrot_vertical.jpg")
    arr = np.array(img)
    new_img = ImageRestoration(arr, l=5, type='vertical').show()