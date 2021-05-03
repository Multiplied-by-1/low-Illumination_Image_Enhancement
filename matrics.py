from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import math

def PSNR(img1:Image, img2:Image):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return psnr(img1, img2)

def SSIM(img1:Image, img2:Image):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return ssim(img1, img2, multichannel=True)

def MAE(img1:Image, img2:Image):
    img1 = np.array(img1)
    img2 = np.array(img2)
    return np.mean(abs(img1 - img2))


def Entrpoy(img: Image): # return[[R(bits, per_pixel)],[G],[B],[L]]
    rgb = img.convert("RGB")
    grey = img.convert("L")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    l = np.asarray(grey)

    def cal_entropy(channel: np.ndarray):  # calculate the entropy for one channel
        hist, _ = np.histogram(channel, bins=range(0, 256))
        hist = hist[hist > 0]
        bits = -np.log10(hist / hist.sum()).sum()
        per_pixel = bits / channel.size
        return [bits, per_pixel]

    return [cal_entropy(r), cal_entropy(g), cal_entropy(b), cal_entropy(l)]

def EME(img: Image, s: int): #s: the size of small square batch to be calculated
    rgb = img.convert("RGB")
    grey = img.convert("L")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    l = np.asarray(grey)

    def cal_eme(channel: np.ndarray, s: int):
        s1, s2 = channel.shape
        s1 = math.floor(s1 / s)
        s2 = math.floor(s2 / s)
        eme = 0
        for i in range(0, channel.shape[0], s1):
            for j in range(0, channel.shape[1], s2):
                temp_c = channel[i:i+s1, j:j+s2]
                eme += 20 * np.log10((np.max(temp_c) + 1) / (np.min(temp_c) + 1)) #using 1e-6 might cause the dark image's EME to be very large
        return eme / (s1 * s2)

    return [cal_eme(r, s), cal_eme(g, s), cal_eme(b, s), cal_eme(l, s)]

