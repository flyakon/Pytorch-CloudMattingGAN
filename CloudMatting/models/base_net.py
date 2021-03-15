'''
@anthor: Wenyuan Li
@desc: Networks for base
@date: 2020/8/5
'''

import torch.nn as nn

class BaseNet(nn.Module):

    def print_key_args(self, **kwargs):
        for key, value in kwargs.items():
            print('{0}:{1}'.format(key, value))