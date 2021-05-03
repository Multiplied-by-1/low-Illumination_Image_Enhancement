import argparse
from PIL import Image
from utils import show_results, print_results
from method.HE import HE
from method.Gamma import Gamma
from method.Gray_World import Gray_World
from method.Retinex import Retinex
from method.Max_RGB import Max_RGB
from method.DeHaze import DeHaze
from method.LIME import LIME

def parse_args(parser):
    parser.add_argument('--method', type=str, default='HE', help='The method you would like to apply', choices=['HE', 'Gamma', 'Gray_World', 'Retinex', 'Max_RGB', 'DeHaze', 'LIME'])
    parser.add_argument('--img_path', type=str, default='./tasks/1.bmp', help='Path of the image to be enhanced')
    parser.add_argument('--has_reference', type=str, default='False', help='Does the image has a ground truth reference? If true, enter the path of the reference image')
    parser.add_argument('--eme_size', type=int, default=20, help='The window size of matrix EME')
    return

def method_required(parser):
    if parser.parse_known_args()[0].method in ['Gamma', 'Max_RGB']:
        parser.add_argument('--gamma', type=float, default=0.5, help='Gamma < 1')
    if parser.parse_known_args()[0].method in 'Gray_World':
        parser.add_argument('--gamma', type=float, default=0.5, help='Gamma < 1 for low illumination images, if gamma==1, simply apply Gray World')
    if parser.parse_known_args()[0].method == 'Retinex':
        parser.add_argument('--model', type=str, default='SSR', help='Single or Multiple', choices=['SSR','MSR'])
        parser.add_argument('--sigma_list', type=list, default=[15, 80, 200], help='Sigma list for MSR')
        parser.add_argument('--sigma', type=int, default=200, help='Sigma for SSR')
    if parser.parse_known_args()[0].method == 'DeHaze':
        parser.add_argument('--omega', type=float, default=0.8, help='weight of dark channel')
        parser.add_argument('--kernel_size', type=int, default=9, help='dark channel kernel size')
        parser.add_argument('--model', type=str, default='origin', help='choose to use the original dehaze model of the enhanced one', choices=['origin', 'enhanced'])
    if parser.parse_known_args()[0].method == 'LIME':
        parser.add_argument('--gamma', type=float, default=0.6, help='Gamma < 1')
        parser.add_argument('--kernel_size', type=int, default=15, help='kernel size of special guassian kernel')
        parser.add_argument('--sigma', type=float, default=3, help='sigma for building special gaussian kernel')
        parser.add_argument('--alpha', type=float, default=0.15, help='coefficient to balance the terms in equation 18')
    return

def main(args):
    img = Image.open(args.img_path)
    if args.has_reference != 'False':
        ref_img = Image.open(args.has_reference)

    #begin processing
    if args.method == 'HE':
        model = HE()
    if args.method == 'Gamma':
        model = Gamma(args.gamma)
    if args.method == 'Gray_World':
        model = Gray_World(args.gamma)
    if args.method == 'Retinex':
        model = Retinex(args)
    if args.method == 'Max_RGB':
        model = Max_RGB(args.gamma)
    if args.method == 'DeHaze':
        model = DeHaze(args.omega, args.kernel_size, args.model)
    if args.method == 'LIME':
        model =LIME(args.gamma, args.alpha, args.sigma, args.kernel_size)

    pro_img = model.run(img) #processing image

    #print out the results
    if args.has_reference == 'False':
        show_results(img, pro_img)
        print_results(args, img, pro_img)
    else:
        show_results(img, pro_img, ref_img)
        print_results(args, img, pro_img, ref_img)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Low Illumination Image Enhancement')
    parse_args(parser)
    method_required(parser)
    args = parser.parse_known_args()[0]
    main(args)
