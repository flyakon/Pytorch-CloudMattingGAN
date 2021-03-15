from .matting_transforms import Rotate,HorizontalFlip,VerticalFlip,ColorJitter,Resize,ToTensor,RandomCrop
import torchvision

transforms_dict={
                 'RandomHorizontalFlip':HorizontalFlip,
                 'RandomVerticalFlip':VerticalFlip,
                 'Rotate':Rotate,
                 'ColorJitter':ColorJitter,
                 'Resize':Resize,
                 'ToTensor':ToTensor,
                  'RandomCrop':RandomCrop}


def build_transforms(name='RandomCrop',**kwargs):
    if name in transforms_dict.keys():
        return transforms_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in available values.'.format(name))