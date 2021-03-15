import torch
import torch.nn as nn
import torchvision
from ..base_net import BaseNet
from ..backbone.builder import build_backbone
import torch.utils.data as data_utils
import numpy as np

class Generator(nn.Module):
    def __init__(self,
                 backbone_cfg: dict,
                 head_cfg: dict, **kwargs):
        super(Generator, self).__init__()
        self.backbone = build_backbone(**backbone_cfg)
        self.build_arch(head_cfg)
        self.backbone_cfg = backbone_cfg
        self.img_size=head_cfg['img_size']
        self.gan_input_size=head_cfg['gan_img_size']


    def build_arch(self, head_cfg):
        in_channels = head_cfg['in_channels']
        feat_channels = head_cfg['feat_channels']

        self.trans_conv1 = nn.Sequential(*[
            nn.Conv2d(in_channels, feat_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[0]),
            nn.ReLU(inplace=True)])
        self.trans_conv2 = nn.Sequential(*[
            nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[1]),
            nn.ReLU(inplace=True)])
        self.trans_conv3 = nn.Sequential(*[
            nn.Conv2d(feat_channels[1], feat_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True)])
        self.trans_conv4 = nn.Sequential(*[
            nn.Conv2d(feat_channels[2], feat_channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[3]),
            nn.ReLU(inplace=True)])
        self.trans_conv5 = nn.Sequential(*[
            nn.Conv2d(feat_channels[3], feat_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[4]),
            nn.ReLU(inplace=True)])

        self.genarated_reflectance = nn.Conv2d(feat_channels[4], 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid=nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, bg_img,thick_img, **kwargs):
        input_bg=torch.nn.functional.interpolate(bg_img,size=(self.gan_input_size,self.gan_input_size),
                                                 align_corners=True,mode='bilinear')
        input_thick = torch.nn.functional.interpolate(thick_img, size=(self.gan_input_size, self.gan_input_size),
                                                   align_corners=True, mode='bilinear')
        x=torch.cat((input_bg,input_thick),dim=1)
        x, endpoints = self.backbone(x)

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv1(x)
        if 'block5' in endpoints.keys():
            x = x + endpoints['block5']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv2(x)
        if 'block4' in endpoints.keys():
            x = x + endpoints['block4']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv3(x)
        if 'block3' in endpoints.keys():
            x = x + endpoints['block3']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv4(x)
        if 'block2' in endpoints.keys():
            x = x + endpoints['block2']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv5(x)
        if 'block1' in endpoints.keys():
            x = x + endpoints['block1']

        generated_reflectance = self.genarated_reflectance(x)
        generated_reflectance=self.sigmoid(generated_reflectance)
        # generated_reflectance=self.relu(generated_reflectance)
        generated_reflectance = torch.nn.functional.interpolate(generated_reflectance, align_corners=True,
                                                                size=(self.img_size, self.img_size), mode='bilinear')
        generated_reflectance = generated_reflectance * thick_img*2
        # generated_reflectance = torch.clamp(generated_reflectance, 0., 1.)

        f = torch.from_numpy(np.random.uniform(0, 1, [generated_reflectance.shape[0], 1, 1, 1])).to(
            generated_reflectance.device)
        f = f.float()
        generated_alpha = generated_reflectance * \
                          (1. + 0.2 * f)
        generated_alpha = torch.clamp(generated_alpha, 0., 1.)
        return generated_alpha, generated_reflectance




class Discrimator(nn.Module):
    def __init__(self,
                 backbone_cfg: dict,
                 head_cfg: dict, **kwargs):
        super(Discrimator, self).__init__()
        self.backbone = build_backbone(**backbone_cfg)
        self.build_arch(head_cfg)
        self.backbone_cfg = backbone_cfg
        self.img_size=head_cfg['img_size']


    def build_arch(self, head_cfg):
        in_channels = head_cfg['in_channels']
        num_classes = head_cfg['num_classes']
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.class_fc = nn.Sequential(*[
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512,num_classes)
        ])


    def forward(self, x, **kwargs):
        x=torch.nn.functional.interpolate(x,size=(self.img_size,self.img_size),
                                          align_corners=True,mode='bilinear')
        x, _ = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.class_fc(x)
        prob = self.softmax(logits)
        return logits, prob




