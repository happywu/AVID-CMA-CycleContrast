# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torchvision.models as models
from models.network_blocks import BasicR2P1DBlock

# from mmaction.models.backbones.resnet import ResNet


class R2Plus1D(nn.Module):
    """
    Adapted from https://github.com/facebookresearch/VMZ/blob/4c14ee6f8eae8e2ac97fc4c05713b8a112eb1f28/lib/models/video_model.py
    Adaptation has a full Conv3D stem, and does not adjust for the number of dimensions between the spatial and temporal convolution.
    """
    def __init__(self, depth=18):
        super(R2Plus1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        if depth == 10:
            self.conv2x = BasicR2P1DBlock(64, 64)
            self.conv3x = BasicR2P1DBlock(64, 128, stride=(2, 2, 2))
            self.conv4x = BasicR2P1DBlock(128, 256, stride=(2, 2, 2))
            self.conv5x = BasicR2P1DBlock(256, 512, stride=(2, 2, 2))
        elif depth == 18:
            self.conv2x = nn.Sequential(BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64))
            self.conv3x = nn.Sequential(BasicR2P1DBlock(64, 128, stride=(2, 2, 2)), BasicR2P1DBlock(128, 128))
            self.conv4x = nn.Sequential(BasicR2P1DBlock(128, 256, stride=(2, 2, 2)), BasicR2P1DBlock(256, 256))
            self.conv5x = nn.Sequential(BasicR2P1DBlock(256, 512, stride=(2, 2, 2)), BasicR2P1DBlock(512, 512))
        elif depth == 34:
            self.conv2x = nn.Sequential(BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64), BasicR2P1DBlock(64, 64))
            self.conv3x = nn.Sequential(BasicR2P1DBlock(64, 128, stride=(2, 2, 2)), BasicR2P1DBlock(128, 128), BasicR2P1DBlock(128, 128), BasicR2P1DBlock(128, 128))
            self.conv4x = nn.Sequential(BasicR2P1DBlock(128, 256, stride=(2, 2, 2)), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256), BasicR2P1DBlock(256, 256))
            self.conv5x = nn.Sequential(BasicR2P1DBlock(256, 512, stride=(2, 2, 2)), BasicR2P1DBlock(512, 512), BasicR2P1DBlock(512, 512))
        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.out_dim = 512

    def forward(self, x, return_embs=False):
        x_c1 = self.conv1(x)
        x_b1 = self.conv2x(x_c1)
        x_b2 = self.conv3x(x_b1)
        x_b3 = self.conv4x(x_b2)
        x_b4 = self.conv5x(x_b3)
        x_pool = self.pool(x_b4)
        if return_embs:
            return {'conv1': x_c1, 'conv2x': x_b1, 'conv3x': x_b2, 'conv4x': x_b3, 'conv5x': x_b4, 'pool': x_pool}
        else:
            return x_pool

# class VideoR2D(ResNet):
#     def __init__(self, args):
#         super(VideoR2D, self).__init__(
#             depth=50 if args.arch == 'r50' else 18,
#             torchvision_pretrain=False,
#             in_channels=3,
#             out_indices=[3],  # 1: conv-1, x: stage-(x+1)
#             norm_cfg=dict(type='BN')
#         )
#         self.out_dim = 2048 if args.arch == 'r50' else 512
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#
#     def forward(self, x, return_embs=False):
#         assert len(x.shape) == 5
#         (B, C, SL, H, W) = x.shape
#         x = x.transpose(1, 2)
#         x = x.reshape(B*SL, C, H, W)
#         feature = super().forward(x)
#         _n, c, h, w = feature.shape
#         # feature = self.pool(feature).squeeze()  # B * N, C
#         feature = feature.reshape(B, SL, c, h, w).transpose(1, 2)
#         feature = self.avgpool(feature)  # B x c
#         if return_embs:
#             return {'pool': feature}
#         else:
#             return feature

import models.resnet_torch as models_torch

class VideoR2DTorch(nn.Module):
    def __init__(self, args):
        super(VideoR2DTorch, self).__init__()
        model = models_torch.__dict__[args.arch](return_before_avgpool=True)
        self.model = model
        self.out_dim = 2048 if args.arch == 'r50' else 512
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, return_embs=False):
        assert len(x.shape) == 5
        (B, C, SL, H, W) = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(B*SL, C, H, W)
        feature = self.model.forward(x)
        _n, c, h, w = feature.shape
        # feature = self.pool(feature).squeeze()  # B * N, C
        feature = feature.reshape(B, SL, c, h, w).transpose(1, 2)
        feature = self.avgpool(feature)  # B x c
        if return_embs:
            return {'pool': feature}
        else:
            return feature


class AV_WrapperR2D(nn.Module):
    def __init__(self, args):
        super(AV_WrapperR2D, self).__init__()
        self.from_moco = args.from_moco
        self.from_mmaction = args.from_mmaction
        if args.from_moco:
            self.video_model = VideoR2DTorch(args)
        elif args.from_mmaction:
            self.video_model = VideoR2D(args)
        else:
            assert False
        self.audio_model = None
        self.out_dim = 2048 if args.arch == 'r50' else 512

    def load_state_dict(self, state):
        if 'state_dict' in state:
            state = state['state_dict']
        if self.from_mmaction:
            output_dict = dict()
            has_backbone = False
            for key, value in state.items():
                if key.startswith('backbone'):
                    output_dict[key[9:]] = value
                    has_backbone = True
            assert has_backbone
            msg = self.video_model.load_state_dict(output_dict, strict=True)
        elif self.from_moco:
            output_dict = parse_moco_checkpoint(state)
            msg = self.video_model.model.load_state_dict(output_dict, strict=False)
        print('load state', msg)

    def forward(self, x):
        pass


def parse_moco_checkpoint(state_dict):
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]

        if k.startswith('module.target_encoder.net') and not k.startswith('module.target_encoder.net.fc'):
            # remove prefix
            state_dict[k[len("module.target_encoder.net."):]] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]

    return state_dict
