from PIL import Image
import numpy as np

class HE():
    def __init__(self, ):
        return

    def run(self, img: Image):
        r, g, b = img.split()
        r, g, b = np.array(r), np.array(g), np.array(b)

        def cal_he(channel: np.ndarray):
            imhist, bins = np.histogram(channel.flatten(), bins=256, density=True)
            cul_dis = imhist.cumsum()
            cul_dis = 255 * cul_dis / cul_dis[-1]
            result = np.interp(channel.flatten(), bins[:-1], cul_dis).reshape(channel.shape)
            return result

        img = np.stack((cal_he(r), cal_he(g), cal_he(b)), axis=2)
        return Image.fromarray(np.uint8(img))
