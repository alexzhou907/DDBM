""" from clean fid """

import os
import platform
import numpy as np
import torch
import cleanfid
from cleanfid.downloads_helper import check_download_url
from .inception_pytorch import InceptionV3
from .inception_torchscript import InceptionV3W


"""
returns a functions that takes an image in range [0,255]
and outputs a feature embedding vector
"""
def feature_extractor(name="torchscript_inception", device=torch.device("cuda"), resize_inside=False, use_dataparallel=True):
    if name == "torchscript_inception":
        path = "./" if platform.system() == "Windows" else "/tmp"
        model = InceptionV3W(path, download=True, resize_inside=resize_inside).to(device)
        model.eval()
        if use_dataparallel:
            model = torch.nn.DataParallel(model)
        def model_fn(x): return model(x)
    elif name == "pytorch_inception":
        model = InceptionV3(output_blocks=[3], resize_input=False).to(device)
        model.eval()
        if use_dataparallel:
            model = torch.nn.DataParallel(model)
        def model_fn(x): return model(x/255)[0].squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"{name} feature extractor not implemented")
    return model_fn


"""
Build a feature extractor for each of the modes
"""
def build_feature_extractor(mode, device=torch.device("cuda"), use_dataparallel=True):
    assert not (mode == 'legacy_pytorch')
    if mode == "legacy_pytorch":
        feat_model = feature_extractor(name="pytorch_inception", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    elif mode == "legacy_tensorflow":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=True, device=device, use_dataparallel=use_dataparallel)
    elif mode == "clean":
        feat_model = feature_extractor(name="torchscript_inception", resize_inside=False, device=device, use_dataparallel=use_dataparallel)
    return feat_model
