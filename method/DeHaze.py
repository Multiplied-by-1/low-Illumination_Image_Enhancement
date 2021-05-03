from PIL import Image
import numpy as np
import cv2

class DeHaze():
    def __init__(self,omega=0.8, k_size=9, model='origin'):
        self.omega = omega
        self.ksize = k_size
        self.model = model
        return

    def atmosphere(self, img: Image): #calculate the atmospheric light in this paper
        r, g, b = img.split()
        r, g, b = np.array(r).flatten(), np.array(g).flatten(), np.array(b).flatten()
        img_np = np.asarray(img, dtype='float')
        dark_c = np.min(img_np, axis=-1)
        first_100 = dark_c.argsort(axis=None)[-100:]
        first_100r, first_100g, first_100b = r[first_100], g[first_100], b[first_100]
        sum_rgb = first_100r + first_100g + first_100b
        choice = np.argmax(sum_rgb)
        return first_100r[choice], first_100g[choice], first_100b[choice] #float

    def dark_channel(self, img_np:np.ndarray):
        dark_c = np.min(img_np, axis=-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.ksize, self.ksize))
        dark = cv2.erode(dark_c, kernel)

        #Image.fromarray(dark).show()
        return dark #np.ndarray

    def transmission(self, img:Image):
        a = self.atmosphere(img)
        img_np = np.asarray(img, dtype='float')
        img_temp = np.zeros_like(img_np)
        img_temp[:, :, 0] = img_np[:, :, 0] / a[0]
        img_temp[:, :, 1] = img_np[:, :, 1] / a[1]
        img_temp[:, :, 2] = img_np[:, :, 2] / a[2]
        #print(img_temp)
        trans = 1 - self.omega * self.dark_channel(img_temp)
        #Image.fromarray(np.uint8(trans * 255)).show()
        #print(trans)
        return trans #np.ndarray 3d

    def cal_p(self, trans: np.ndarray):
        p = np.zeros_like(trans)
        p[trans > 0.5] = 1
        p[trans < 0.5] = trans[trans < 0.5] * 2
        return p

    def run(self, img: Image):
        img_np = np.asarray(img, dtype='float')
        R = 255 - img_np
        rev_img = Image.fromarray(np.uint8(R))
        a = self.atmosphere(rev_img)
        h, w, _ = img_np.shape
        ar, ag, ab = np.ones((h, w)) * a[0], np.ones((h, w)) * a[1], np.ones((h, w)) * a[2]
        A = np.stack((ar, ag, ab), axis=2)
        t = self.transmission(rev_img)
        T = np.stack((t, t, t), axis=2)
        P = self.cal_p(T)
        if self.model == 'origin':
            t[t < 0.1] = 0.1
            return Image.fromarray(np.uint8(255 - ((R - A) / T + A)))
        else:
            return Image.fromarray(np.uint8(255 - ((R - A) / np.multiply(P, T) + A)))



