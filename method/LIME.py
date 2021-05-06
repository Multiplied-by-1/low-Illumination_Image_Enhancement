import numpy as np
from scipy.ndimage.filters import convolve
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from PIL import Image
import cv2

class LIME():
    def __init__(self, gamma=0.6, alpha=0.15, sigma=3, k_size=15):
        self.size = k_size
        self.sigma = sigma
        self.eps = 1e-3
        self.gamma = gamma
        self.alpha = alpha
        return

    def create_kernel(self, sigma: float, size=15):  # follow the equation 23 in the paper, return np.ndarray (size * size)
        kernel = np.zeros((size, size))
        far = np.floor(size / 2)
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(- ((i - far) ** 2 + (j - far) ** 2) / (2 * (sigma ** 2)))
        return kernel

    def weight_init(self, light: np.ndarray, axis: int, kernel: np.ndarray):  # return the intial weight, np.ndarray
        # following strategy 2,3 calculating the gradient
        grad = cv2.Sobel(light, cv2.CV_64F, int(axis == 1), int(axis == 0), ksize=1)
        # equation 22 in the paper
        W = convolve(np.ones_like(light), kernel, mode='constant') / (np.abs(convolve(grad, kernel, mode='constant')) + self.eps)
        # return w used in equation 19
        return W / (np.abs(grad) + self.eps)

    def update_light(self, light: np.ndarray, gamma: float, alpha: float, kernel: np.ndarray):
        wx = self.weight_init(light, axis=1, kernel=kernel).flatten()
        wy = self.weight_init(light, axis=0, kernel=kernel).flatten()

        h, w = light.shape
        # for easy calculation, we flatten the light
        vec_light = light.copy().flatten()

        row, column, data = [], [], []
        for p in range(h * w):
            diag = 0
            if p - w >= 0:
                temp_weight = wy[p - w]
                row.append(p)
                column.append(p - w)
                data.append(-temp_weight)
                diag += temp_weight
            if p + w < h * w:
                temp_weight = wy[p + w]
                row.append(p)
                column.append(p + w)
                data.append(-temp_weight)
                diag += temp_weight
            if p % w != 0:
                temp_weight = wx[p - 1]
                row.append(p)
                column.append(p - 1)
                data.append(-temp_weight)
                diag += temp_weight
            if p % w != w - 1:
                temp_weight = wx[p + 1]
                row.append(p)
                column.append(p + 1)
                data.append(-temp_weight)
                diag += temp_weight

            row.append(p)
            column.append(p)
            data.append(diag)

        # the sum part in equation 19
        fun = csr_matrix((data, (row, column)), shape=(h * w, h * w))
        # solve the light with the linear system
        I = diags([np.ones(h * w)], [0])
        # A * pro_light(linear) = light(linear) --Equation 19
        A = I + alpha * fun
        pro_l = spsolve(csr_matrix(A), vec_light, permc_spec=None, use_umfpack=True).reshape((h, w))
        # gamma correction
        pro_l = np.clip(pro_l, self.eps, 1) ** gamma

        return pro_l

    def run(self, img: Image):
        kernel = self.create_kernel(self.sigma)
        img_np = np.asarray(img, dtype='float') / 255

        #Calculate light
        light = np.max(img_np, axis=-1)
        light = self.update_light(light, self.gamma, self.alpha, kernel)
        light_3d = np.stack((light, light, light), axis=2)

        #Retinex
        pro_img = img_np / light_3d * 255
        pro_img[pro_img>255] = 255
        pro_img[pro_img < 0] = 0
        return Image.fromarray(np.uint8(pro_img))