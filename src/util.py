# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np


def disp2rgb(disp):
    H = disp.shape[0]
    W = disp.shape[1]

    I = disp.flatten()

    map = np.array(
        [
            [0, 0, 0, 114],
            [0, 0, 1, 185],
            [1, 0, 0, 114],
            [1, 0, 1, 174],
            [0, 1, 0, 114],
            [0, 1, 1, 185],
            [1, 1, 0, 114],
            [1, 1, 1, 0],
        ]
    )
    bins = map[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]

    ind = np.minimum(
        np.sum(np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:, None], I.shape[0], axis=1), axis=0), 6
    )
    bins = np.reciprocal(bins)
    cbins = np.append(np.array([[0]]), cbins[:, None])

    I = np.multiply(I - cbins[ind], bins[ind])
    I = np.minimum(
        np.maximum(
            np.multiply(map[ind, 0:3], np.repeat(1 - I[:, None], 3, axis=1))
            + np.multiply(map[ind + 1, 0:3], np.repeat(I[:, None], 3, axis=1)),
            0,
        ),
        1,
    )

    I = np.reshape(I, [H, W, 3]).astype(np.float32)

    return I


def str2bool(bool_input_string):
    if isinstance(bool_input_string, bool):
        return bool_input_string
    if bool_input_string.lower() in ("true"):
        return True
    elif bool_input_string.lower() in ("false"):
        return False
    else:
        raise NameError("Please provide boolean type.")
