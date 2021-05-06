import numpy as np
import cv2
from PIL import Image

class Retinex():
    def __init__(self, args):
        self.sigma_list = args.sigma_list
        self.sigma = args.sigma
        self.appr = args.model
        return

    def move_zero(self, mat:np.ndarray):
        mat[mat == 0] = min(mat[np.nonzero(mat)])
        return mat

    def SSR(self,img:Image, sigma=100):
        '''
        :return: np.ndarray
        '''
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        blur = (cv2.GaussianBlur(self.move_zero(img_cv), (0,0), sigma))
        light = cv2.multiply(cv2.log(img_cv / 255.0), cv2.log(blur / 255.0))
        res = cv2.subtract(cv2.log(img_cv / 255.0), light)
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)

        res = cv2.convertScaleAbs(res)

        return np.asarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), dtype='float')

    def MSR(self, img:Image, sigma_list=[15, 80, 200]):
        img_np = np.asarray(img, dtype='float')
        retinex = np.zeros_like(img_np)
        for sigma in sigma_list:
            retinex += self.SSR(img, sigma)
        retinex = retinex / len(sigma_list)
        return retinex

    def run(self, img: Image):
        if self.appr == 'SSR':
            pro_img = self.SSR(img, self.sigma)
            return Image.fromarray(np.uint8(pro_img))
        else:
            pro_img = self.MSR(img, self.sigma_list)
            return Image.fromarray(np.uint8(pro_img))
