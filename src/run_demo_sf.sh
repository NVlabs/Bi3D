#!/usr/bin/env bash

# GENERATE BINARY DEPTH SEGMENTATIONS AND COMBINE THEM TO GENERATE QUANTIZED DEPTH
CUDA_VISIBLE_DEVICES=0 python run_binary_depth_estimation.py \
    --arch bi3dnet_binary_depth \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_featnethr_arch featextractnethr \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch segrefinenet \
    --featextractnethr_out_planes 16 \
    --segrefinenet_in_planes 17 \
    --segrefinenet_out_planes 8 \
    --crop_height 576 --crop_width 960 \
    --disp_vals 24 36 54 96 144 \
    --img_left  '../data/sf_img_left.jpg' \
    --img_right '../data/sf_img_right.jpg' \
    --pretrained '../model_weights/sf_binary_depth.pth.tar'


# FULL RANGE CONTINOUS DEPTH ESTIMATION WITHOUT 3D REGULARIZATION
CUDA_VISIBLE_DEVICES=0 python run_continuous_depth_estimation.py \
    --arch bi3dnet_continuous_depth_2D \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch disprefinenet \
    --disprefinenet_out_planes 32 \
    --crop_height 576 --crop_width 960 \
    --disp_range_min 0 \
    --disp_range_max 192 \
    --bi3dnet_max_disparity 192 \
    --img_left  '../data/sf_img_left.jpg' \
    --img_right '../data/sf_img_right.jpg' \
    --pretrained '../model_weights/sf_continuous_depth_no_conf_reg.pth.tar'


# SELECTIVE RANGE CONTINOUS DEPTH ESTIMATION WITHOUT 3D REGULARIZATION
CUDA_VISIBLE_DEVICES=0 python run_continuous_depth_estimation.py \
    --arch bi3dnet_continuous_depth_2D \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch disprefinenet \
    --disprefinenet_out_planes 32 \
    --crop_height 576 --crop_width 960 \
    --disp_range_min 18 \
    --disp_range_max 60 \
    --bi3dnet_max_disparity 192 \
    --img_left  '../data/sf_img_left.jpg' \
    --img_right '../data/sf_img_right.jpg' \
    --pretrained '../model_weights/sf_continuous_depth_no_conf_reg.pth.tar'


# FULL RANGE CONTINOUS DEPTH ESTIMATION WITH 3D REGULARIZATION 
CUDA_VISIBLE_DEVICES=0 python run_continuous_depth_estimation.py \
    --arch bi3dnet_continuous_depth_3D \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch disprefinenet \
    --bi3dnet_regnet_arch segregnet3d \
    --disprefinenet_out_planes 32 \
    --regnet_out_planes 16 \
    --crop_height 576 --crop_width 960 \
    --disp_range_min 0 \
    --disp_range_max 192 \
    --bi3dnet_max_disparity 192 \
    --img_left  '../data/sf_img_left.jpg' \
    --img_right '../data/sf_img_right.jpg' \
    --pretrained '../model_weights/sf_continuous_depth_conf_reg.pth.tar'
