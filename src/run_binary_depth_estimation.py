# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import models
import cv2
import numpy as np

from util import disp2rgb, str2bool
import random

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))


# Parse arguments
parser = argparse.ArgumentParser(allow_abbrev=False)

# Model
parser.add_argument("--arch", type=str, default="bi3dnet_binary_depth")

parser.add_argument("--bi3dnet_featnet_arch", type=str, default="featextractnetspp")
parser.add_argument("--bi3dnet_featnethr_arch", type=str, default="featextractnethr")
parser.add_argument("--bi3dnet_segnet_arch", type=str, default="segnet2d")
parser.add_argument("--bi3dnet_refinenet_arch", type=str, default="segrefinenet")
parser.add_argument("--bi3dnet_max_disparity", type=int, default=192)
parser.add_argument("--bi3dnet_disps_per_example_true", type=str2bool, default=True)

parser.add_argument("--featextractnethr_out_planes", type=int, default=16)
parser.add_argument("--segrefinenet_in_planes", type=int, default=17)
parser.add_argument("--segrefinenet_out_planes", type=int, default=8)

# Input
parser.add_argument("--pretrained", type=str)
parser.add_argument("--img_left", type=str)
parser.add_argument("--img_right", type=str)
parser.add_argument("--disp_vals", type=float, nargs="*")
parser.add_argument("--crop_height", type=int)
parser.add_argument("--crop_width", type=int)

args, unknown = parser.parse_known_args()

####################################################################################################
def main():
    """
    Main function.

    Args:
    """

    options = vars(args)
    print("==> ALL PARAMETERS")
    for key in options:
        print("{} : {}".format(key, options[key]))

    out_dir = "out"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    base_name = os.path.splitext(os.path.basename(args.img_left))[0]

    # Model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](options, network_data).cuda()

    # Inputs
    img_left = Image.open(args.img_left).convert("RGB")
    img_left = transforms.functional.to_tensor(img_left)
    img_left = transforms.functional.normalize(img_left, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_left = img_left.type(torch.cuda.FloatTensor)[None, :, :, :]
    img_right = Image.open(args.img_right).convert("RGB")
    img_right = transforms.functional.to_tensor(img_right)
    img_right = transforms.functional.normalize(img_right, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_right = img_right.type(torch.cuda.FloatTensor)[None, :, :, :]

    segs = []
    for disp_val in args.disp_vals:

        assert disp_val % 3 == 0, "disparity value should be a multiple of 3 as we downsample the image by 3"
        disp_long = torch.Tensor([[disp_val / 3]]).type(torch.LongTensor).cuda()

        # Pad inputs
        tw = args.crop_width
        th = args.crop_height
        assert tw % 96 == 0, "image dimensions should be a multiple of 96"
        assert th % 96 == 0, "image dimensions should be a multiple of 96"
        h = img_left.shape[2]
        w = img_left.shape[3]
        x1 = random.randint(0, max(0, w - tw))
        y1 = random.randint(0, max(0, h - th))
        pad_w = tw - w if tw - w > 0 else 0
        pad_h = th - h if th - h > 0 else 0
        pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
        img_left = img_left[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]
        img_right = img_right[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]
        img_left_pad = pad_opr(img_left)
        img_right_pad = pad_opr(img_right)

        # Inference
        model.eval()
        with torch.no_grad():
            output = model(img_left_pad, img_right_pad, disp_long)[1][:, :, pad_h:, pad_w:]

        # Write binary depth results
        seg_img = output[0, 0][None, :, :].clone().cpu().detach().numpy()
        seg_img = np.transpose(seg_img * 255.0, (1, 2, 0))
        cv2.imwrite(
            os.path.join(out_dir, "%s_%s_seg_confidence_%d.png" % (base_name, args.arch, disp_val)), seg_img
        )

        segs.append(output[0, 0][None, :, :].clone().cpu().detach().numpy())

    # Generate quantized depth results
    segs = np.concatenate(segs, axis=0)
    segs = np.insert(segs, 0, np.ones((1, h, w), dtype=np.float32), axis=0)
    segs = np.append(segs, np.zeros((1, h, w), dtype=np.float32), axis=0)

    segs = 1.0 - segs

    # Get the pdf values for each segmented region
    pdf_method = segs[1:, :, :] - segs[:-1, :, :]

    # Get the labels
    labels_method = np.argmax(pdf_method, axis=0).astype(np.int)
    disp_map = labels_method.astype(np.float32)

    disp_vals = args.disp_vals
    disp_vals.insert(0, 0)
    disp_vals.append(args.bi3dnet_max_disparity)

    for i in range(len(disp_vals) - 1):
        min_disp = disp_vals[i]
        max_disp = disp_vals[i + 1]
        mid_disp = 0.5 * (min_disp + max_disp)
        disp_map[labels_method == i] = mid_disp

    disp_vals_str_list = ["%d" % disp_val for disp_val in disp_vals]
    disp_vals_str = "-".join(disp_vals_str_list)

    img_disp = np.clip(disp_map, 0, args.bi3dnet_max_disparity)
    img_disp = img_disp / args.bi3dnet_max_disparity
    img_disp = (disp2rgb(img_disp) * 255.0).astype(np.uint8)

    cv2.imwrite(
        os.path.join(out_dir, "%s_%s_quant_depth_%s.png" % (base_name, args.arch, disp_vals_str)), img_disp
    )

    return


if __name__ == "__main__":
    main()
