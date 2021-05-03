from PIL import Image
import numpy as np

class Max_RGB():
    def __init__(self, gamma=0.6):
        self.gamma = gamma
        return

    def move_zero(self, mat:np.ndarray):
        mat[mat == 0] = min(mat[np.nonzero(mat)])
        return mat

    def run(self, img: Image):

        img_np = np.asarray(img, dtype='float')/255
        img_np = self.move_zero(img_np)
        light = np.max(img_np, axis=-1)
        light = light ** self.gamma
        light_3d = np.stack((light, light, light), axis=2)
        pro_img = img_np / light_3d

        return Image.fromarray(np.uint8(pro_img * 255))