from skimage import exposure
from PIL import Image
import numpy as np

class Gamma():
    def __init__(self, gamma):
        self.gamma = gamma
        return

    def run(self, img: Image):
        img_np = np.asarray(img)
        img_pro = exposure.adjust_gamma(img_np, gamma=self.gamma, gain=1)
        return Image.fromarray(img_pro)