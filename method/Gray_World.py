from PIL import Image
import numpy as np
from method.Gamma import Gamma

class Gray_World():
    def __init__(self, gamma):
        self.gamma = gamma
        return

    def run(self, img: Image):
        corr = Gamma(self.gamma)
        img = corr.run(img)
        r, g, b = img.split()
        r, g, b = np.array(r), np.array(g), np.array(b)
        avg_r, avg_g, avg_b = np.average(r), np.average(g), np.average(b)
        avg = (avg_r + avg_g + avg_b) / 3
        pro_r = r * avg / avg_r
        pro_g = g * avg / avg_g
        pro_b = b * avg / avg_b
        img = np.stack((pro_r, pro_g, pro_b), axis=2)
        img[img>255] = 255

        return Image.fromarray(np.uint8(img))
