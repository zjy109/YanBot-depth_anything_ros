import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time

from depth_anything_v2.dpt import DepthAnythingV2

def convert_to_absolute_depth(estimated_depth, measured_depth):
    D = measured_depth.astype(np.float32)
    X = estimated_depth.astype(np.float32)

    valid_mask = D > 0
    D_valid = D[valid_mask]
    X_valid = X[valid_mask]

    X_stack = np.vstack([X_valid, np.ones_like(X_valid)]).T
    params, residuals, rank, s = np.linalg.lstsq(X_stack, D_valid, rcond=None)
    A, b = params

    print(f"Scale factor (A): {A}, Offset (b): {b}")


    absolute_depth = A * estimated_depth + b
    return absolute_depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--img-path', type=str, required=False)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'../weights/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # os.makedirs(args.outdir, exist_ok=True)

    # if os.path.isfile(args.img_path):
    #     if args.img_path.endswith('txt'):
    #         with open(args.img_path, 'r') as f:
    #             filenames = f.read().splitlines()
    #     else:
    #         filenames = [args.img_path]
    # else:
    #     filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    # color_filename = '1741221825.671469450_color.png'
    # depth_filename = '1741221825.671469450_depth.png'

    # raw_image = cv2.imread(color_filename)
    # print(f'Loaded image: {color_filename}')

    filenames_color=glob.glob(os.path.join("./test_image_color/", '**/*'), recursive=True)
    filenames_depth=glob.glob(os.path.join("./test_image_depth/", '**/*'), recursive=True)

    for filename_color, filename_depth in zip(filenames_color,filenames_depth):
        raw_image = cv2.imread(filename_color)

        time.sleep(0.9)

        T1 = time.time()

        estimated_depth = depth_anything.infer_image(raw_image, args.input_size)

        T2 = time.time()
        print('深度估计用时:%s毫秒' % ((T2 - T1)*1000))
        # T1=T2

        measured_depth = cv2.imread(filename_depth, cv2.IMREAD_UNCHANGED)
        # print(f'Loaded depth: {depth_filename}')


        absolute_depth = convert_to_absolute_depth(estimated_depth, measured_depth)

        T2 = time.time()
        print('总用时:%s毫秒\n' % ((T2 - T1)*1000))


    # vis_depth = cv2.convertScaleAbs(absolute_depth, alpha=0.1)


    # abs_filename = os.path.join(args.outdir, 'absolute_depth.png')
    # vis_filename = os.path.join(args.outdir, 'vis_depth.png')

    # cv2.imwrite(abs_filename, absolute_depth.astype(np.uint16))
    # cv2.imwrite(vis_filename, vis_depth)

    # print(f'Saved absolute depth: {abs_filename}')
    # print(f'Saved visualization depth: {vis_filename}')
