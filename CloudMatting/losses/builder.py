import torch
import torchvision
import torch.nn as nn
from .gan_loss import VanillaLoss,LSGANLoss,WGANLosss

losses_dict={'CrossEntropyLoss':nn.CrossEntropyLoss,
             'BCEWithLogitsLoss':nn.BCEWithLogitsLoss,
             'L1Loss':nn.L1Loss,
             'VanhingenLoss':VanillaLoss,
             'LSGANLoss':LSGANLoss,
             'WGANLoss':WGANLosss}


def builder_loss(name='CrossEntropyLoss',**kwargs):

    if name in losses_dict.keys():
        return losses_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))