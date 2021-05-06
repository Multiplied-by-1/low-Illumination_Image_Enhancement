from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matrics import PSNR, SSIM, MAE, Entrpoy, EME

def draw_hist(img: Image, plt):
    r, g, b = img.split()
    ar=np.array(r).flatten()
    plt.hist(ar, bins=256, density=1, alpha=0.5, stacked=True, facecolor='r',edgecolor='r')
    ag=np.array(g).flatten()
    plt.hist(ag, bins=256, density=1, alpha=0.5, stacked=True, facecolor='g',edgecolor='g')
    ab=np.array(b).flatten()
    plt.hist(ab, bins=256, density=1, alpha=0.5, stacked=True, facecolor='b',edgecolor='b')

def show_results(img: Image, pro_img: Image, ref_img=None):
    if ref_img == None:
        plt.subplot(2 ,2 ,1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(pro_img)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        draw_hist(img, plt)
        plt.subplot(2, 2, 4)
        draw_hist(pro_img, plt)

    else:
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(pro_img)
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(ref_img)
        plt.axis('off')
        plt.subplot(2, 3, 4)
        draw_hist(img, plt)
        plt.subplot(2, 3, 5)
        draw_hist(pro_img, plt)
        plt.subplot(2, 3, 6)
        draw_hist(ref_img, plt)

    plt.show()
    return

def print_results(args, img: Image, pro_img: Image, ref_img=None):
    print('=' * 100)
    print('Finish Processing! Method:', args.method)
    print('=' * 100)
    img_entropy = Entrpoy(img)
    pro_entropy = Entrpoy(pro_img)
    img_eme = EME(img, args.eme_size)
    pro_eme = EME(pro_img, args.eme_size)

    print('Entropy of RGB channels of the original image:', img_entropy[0][0], img_entropy[1][0], img_entropy[2][0])
    print('Entropy of RGB channels of the enhanced image:', pro_entropy[0][0], pro_entropy[1][0], pro_entropy[2][0], '\n')
    print('Entropy of the Grayscale original image:', img_entropy[3][0])
    print('Entropy of the Grayscale enhanced image:', pro_entropy[3][0],'\n')

    print('EME of RGB channels of the original image:', img_eme[0], img_eme[1], img_eme[2])
    print('EME of RGB channels of the enhanced image:', pro_eme[0], pro_eme[1], pro_eme[2],'\n')
    print('EME of the Grayscale original image:', img_eme[3])
    print('EME of the Grayscale enhanced image:', pro_eme[3], '\n')

    if ref_img != None:
        print('PSNR between the original image and the ground truth images is', PSNR(img, ref_img))
        print('PSNR between the enhanced image and the ground truth images is', PSNR(pro_img, ref_img),'\n')
        print('SSIM between the original image and the ground truth images is', SSIM(img, ref_img))
        print('SSIM between the enhanced image and the ground truth images is', SSIM(pro_img, ref_img),'\n')
        print('MAE between the original image and the ground truth images is', MAE(img, ref_img))
        print('MAE between the enhanced image and the ground truth images is', MAE(pro_img, ref_img),'\n')
    print('=' * 100)