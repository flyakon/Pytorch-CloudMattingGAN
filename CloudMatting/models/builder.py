from .matting.cloud_gan_matting_net import CloudGANMattingNet

models_dict={'CloudGANMattingNet':CloudGANMattingNet,
             }

def builder_models(name='CloudGANMattingNet',**kwargs):
    if name in models_dict.keys():
        return models_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))