# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import models.FeatExtractNet as FeatNet
import models.SegNet2D as SegNet
import models.RefineNet2D as RefineNet
import models.RefineNet3D as RefineNet3D


__all__ = ["bi3dnet_binary_depth", "bi3dnet_continuous_depth_2D", "bi3dnet_continuous_depth_3D"]


def compute_cost_volume(features_left, features_right, disp_ids, max_disp, is_disps_per_example):

    batch_size = features_left.shape[0]
    feature_size = features_left.shape[1]
    H = features_left.shape[2]
    W = features_left.shape[3]

    psv_size = disp_ids.shape[1]

    psv = Variable(features_left.new_zeros(batch_size, psv_size, feature_size * 2, H, W + max_disp)).cuda()

    if is_disps_per_example:
        for i in range(batch_size):
            psv[i, 0, :feature_size, :, 0:W] = features_left[i]
            psv[i, 0, feature_size:, :, disp_ids[i, 0] : W + disp_ids[i, 0]] = features_right[i]
        psv = psv.contiguous()
    else:
        for i in range(psv_size):
            psv[:, i, :feature_size, :, 0:W] = features_left
            psv[:, i, feature_size:, :, disp_ids[0, i] : W + disp_ids[0, i]] = features_right
        psv = psv.contiguous()

    return psv


"""
Bi3DNet for continuous depthmap generation. Doesn't use 3D regularization.
"""


class Bi3DNetContinuousDepth2D(nn.Module):
    def __init__(self, options, featnet_arch, segnet_arch, refinenet_arch=None, max_disparity=192):

        super(Bi3DNetContinuousDepth2D, self).__init__()

        self.max_disparity = max_disparity
        self.max_disparity_seg = int(self.max_disparity / 3)
        self.is_disps_per_example = False
        self.is_save_memory = False

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False

        self.featnet = FeatNet.__dict__[featnet_arch](options, data=None)
        self.segnet = SegNet.__dict__[segnet_arch](options, data=None)
        if self.is_refine:
            self.refinenet = RefineNet.__dict__[refinenet_arch](options, data=None)

        return

    def forward(self, img_left, img_right, disp_ids):

        batch_size = img_left.shape[0]
        psv_size = disp_ids.shape[1]

        if psv_size == 1:
            self.is_disps_per_example = True
        else:
            self.is_disps_per_example = False

        # Feature Extraction
        features_left = self.featnet(img_left)
        features_right = self.featnet(img_right)
        feature_size = features_left.shape[1]
        H = features_left.shape[2]
        W = features_left.shape[3]

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, disp_ids, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]
        seg_raw_low_res = seg_raw_low_res.view(batch_size, 1, psv_size, H, W)

        # Upsampling
        seg_prob_low_res_up = torch.sigmoid(
            F.interpolate(
                seg_raw_low_res,
                size=[psv_size * 3, img_left.size()[-2], img_left.size()[-1]],
                mode="trilinear",
                align_corners=False,
            )
        )
        seg_prob_low_res_up = seg_prob_low_res_up[:, 0, 1:-1, :, :]

        # Projection
        disparity_normalized = torch.mean((seg_prob_low_res_up), dim=1, keepdim=True)

        # Refinement
        if self.is_refine:
            refine_net_input = torch.cat((disparity_normalized, img_left), dim=1)
            disparity_normalized = self.refinenet(refine_net_input)

        return seg_prob_low_res_up, disparity_normalized


def bi3dnet_continuous_depth_2D(options, data=None):

    print("==> USING Bi3DNetContinuousDepth2D")
    for key in options:
        if "bi3dnet" in key:
            print("{} : {}".format(key, options[key]))

    model = Bi3DNetContinuousDepth2D(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
    )

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model


"""
Bi3DNet for continuous depthmap generation. Uses 3D regularization.
"""


class Bi3DNetContinuousDepth3D(nn.Module):
    def __init__(
        self,
        options,
        featnet_arch,
        segnet_arch,
        refinenet_arch=None,
        refinenet3d_arch=None,
        max_disparity=192,
    ):

        super(Bi3DNetContinuousDepth3D, self).__init__()

        self.max_disparity = max_disparity
        self.max_disparity_seg = int(self.max_disparity / 3)
        self.is_disps_per_example = False
        self.is_save_memory = False

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False

        self.featnet = FeatNet.__dict__[featnet_arch](options, data=None)
        self.segnet = SegNet.__dict__[segnet_arch](options, data=None)
        if self.is_refine:
            self.refinenet = RefineNet.__dict__[refinenet_arch](options, data=None)
            self.refinenet3d = RefineNet3D.__dict__[refinenet3d_arch](options, data=None)

        return

    def forward(self, img_left, img_right, disp_ids):

        batch_size = img_left.shape[0]
        psv_size = disp_ids.shape[1]

        if psv_size == 1:
            self.is_disps_per_example = True
        else:
            self.is_disps_per_example = False

        # Feature Extraction
        features_left = self.featnet(img_left)
        features_right = self.featnet(img_right)
        feature_size = features_left.shape[1]
        H = features_left.shape[2]
        W = features_left.shape[3]

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, disp_ids, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]  # cropped to remove excess boundary
        seg_raw_low_res = seg_raw_low_res.view(batch_size, 1, psv_size, H, W)

        # Upsampling
        seg_prob_low_res_up = torch.sigmoid(
            F.interpolate(
                seg_raw_low_res,
                size=[psv_size * 3, img_left.size()[-2], img_left.size()[-1]],
                mode="trilinear",
                align_corners=False,
            )
        )

        seg_prob_low_res_up = seg_prob_low_res_up[:, 0, 1:-1, :, :]

        # Upsampling after 3D Regularization
        seg_raw_low_res_refined = seg_raw_low_res
        seg_raw_low_res_refined[:, :, 1:, :, :] = self.refinenet3d(
            features_left, seg_raw_low_res_refined[:, :, 1:, :, :]
        )

        seg_prob_low_res_refined_up = torch.sigmoid(
            F.interpolate(
                seg_raw_low_res_refined,
                size=[psv_size * 3, img_left.size()[-2], img_left.size()[-1]],
                mode="trilinear",
                align_corners=False,
            )
        )

        seg_prob_low_res_refined_up = seg_prob_low_res_refined_up[:, 0, 1:-1, :, :]

        # Projection
        disparity_normalized_noisy = torch.mean((seg_prob_low_res_refined_up), dim=1, keepdim=True)

        # Refinement
        if self.is_refine:
            refine_net_input = torch.cat((disparity_normalized_noisy, img_left), dim=1)
            disparity_normalized = self.refinenet(refine_net_input)

        return (
            seg_prob_low_res_up,
            seg_prob_low_res_refined_up,
            disparity_normalized_noisy,
            disparity_normalized,
        )


def bi3dnet_continuous_depth_3D(options, data=None):

    print("==> USING Bi3DNetContinuousDepth3D")
    for key in options:
        if "bi3dnet" in key:
            print("{} : {}".format(key, options[key]))

    model = Bi3DNetContinuousDepth3D(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        refinenet3d_arch=options["bi3dnet_regnet_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
    )

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model


"""
Bi3DNet for binary depthmap generation.
"""


class Bi3DNetBinaryDepth(nn.Module):
    def __init__(
        self,
        options,
        featnet_arch,
        segnet_arch,
        refinenet_arch=None,
        featnethr_arch=None,
        max_disparity=192,
        is_disps_per_example=False,
    ):

        super(Bi3DNetBinaryDepth, self).__init__()

        self.max_disparity = max_disparity
        self.max_disparity_seg = int(max_disparity / 3)
        self.is_disps_per_example = is_disps_per_example

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False

        self.featnet = FeatNet.__dict__[featnet_arch](options, data=None)
        self.featnethr = FeatNet.__dict__[featnethr_arch](options, data=None)
        self.segnet = SegNet.__dict__[segnet_arch](options, data=None)
        if self.is_refine:
            self.refinenet = RefineNet.__dict__[refinenet_arch](options, data=None)

        return

    def forward(self, img_left, img_right, disp_ids):

        batch_size = img_left.shape[0]
        psv_size = disp_ids.shape[1]

        if psv_size == 1:
            self.is_disps_per_example = True
        else:
            self.is_disps_per_example = False

        # Feature Extraction
        features = self.featnet(torch.cat((img_left, img_right), dim=0))

        features_left = features[:batch_size, :, :, :]
        features_right = features[batch_size:, :, :, :]

        if self.is_refine:
            features_lefthr = self.featnethr(img_left)
        feature_size = features_left.shape[1]
        H = features_left.shape[2]
        W = features_left.shape[3]

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, disp_ids, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]  # cropped to remove excess boundary
        seg_prob_low_res = torch.sigmoid(seg_raw_low_res)
        seg_prob_low_res = seg_prob_low_res.view(batch_size, psv_size, H, W)

        seg_prob_low_res_up = F.interpolate(
            seg_prob_low_res, size=img_left.size()[-2:], mode="bilinear", align_corners=False
        )
        out = []
        out.append(seg_prob_low_res_up)

        # Refinement
        if self.is_refine:
            seg_raw_high_res = F.interpolate(
                seg_raw_low_res, size=img_left.size()[-2:], mode="bilinear", align_corners=False
            )
            # Refine Net
            features_left_expand = (
                features_lefthr[:, None, :, :, :].expand(-1, psv_size, -1, -1, -1).contiguous()
            )
            features_left_expand = features_left_expand.view(
                -1, features_lefthr.size()[1], features_lefthr.size()[2], features_lefthr.size()[3]
            )
            refine_net_input = torch.cat((seg_raw_high_res, features_left_expand), dim=1)

            seg_raw_high_res = self.refinenet(refine_net_input)

            seg_prob_high_res = torch.sigmoid(seg_raw_high_res)
            seg_prob_high_res = seg_prob_high_res.view(
                batch_size, psv_size, img_left.size()[-2], img_left.size()[-1]
            )
            out.append(seg_prob_high_res)
        else:
            out.append(seg_prob_low_res_up)

        return out


def bi3dnet_binary_depth(options, data=None):

    print("==> USING Bi3DNetBinaryDepth")
    for key in options:
        if "bi3dnet" in key:
            print("{} : {}".format(key, options[key]))

    model = Bi3DNetBinaryDepth(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        featnethr_arch=options["bi3dnet_featnethr_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
        is_disps_per_example=options["bi3dnet_disps_per_example_true"],
    )

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model
