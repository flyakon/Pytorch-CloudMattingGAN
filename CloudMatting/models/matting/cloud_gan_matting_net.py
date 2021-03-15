
'''
@anthor: Wenyuan Li
@desc: Networks for Cloud matting
@date: 2020/9/12
'''
import torch
import torchvision
import numpy as np
import torch.nn as nn
from ..base_net import BaseNet
from ..backbone.builder import build_backbone
import torch.utils.data as data_utils
from CloudMatting.dataset.cloud_matting_dataset import CloudMattingDataset,CloudSingleDataset
from CloudMatting.models.matting.cloud_matting_net import CloudMattingNet
from .genenrator_discrimator_net import Generator,Discrimator


import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from CloudMatting.losses.builder import builder_loss
from CloudMatting.utils.optims.builder import build_lr_schedule,build_optim
import CloudMatting.utils.utils as utils
from CloudMatting.metric.time_metric import TimeMetric
import tqdm

class CloudGANMattingNet(BaseNet):
    def __init__(self,
                 generator_cfg:dict,
                 discrimator_cfg:dict,
                 train_cfg:dict,
                 test_cfg:dict,with_matting=True,**kwargs):
        super(CloudGANMattingNet,self).__init__()
        self.generator=Generator(**generator_cfg)
        self.discrimator=Discrimator(**discrimator_cfg)
        self.with_matting=with_matting
        if self.with_matting:
            if 'matting_cfg' in kwargs.keys() :
                print('build matting net')
                self.with_matting=True
                self.matting_model=CloudMattingNet(**kwargs['matting_cfg'],train_cfg=train_cfg,test_cfg=test_cfg)
            else:
                raise NotImplementedError('set with_matting = True, but find no matting cfg.')
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def forward(self, bg_img,thick_img,cloud_img, **kwargs):

        generated_alpha,generated_reflectance=self.generator.forward(bg_img,thick_img)
        generated_img= generated_reflectance+(1-generated_alpha)*bg_img

        fake_logits,_=self.discrimator(generated_img)
        real_logits,_=self.discrimator(cloud_img)
        return generated_reflectance,generated_alpha,generated_img,fake_logits,real_logits


    def run_train_interface(self,**kwargs):
        if self.with_matting:
            self.run_train_matting(**kwargs)
        else:
            self.run_train_generation(**kwargs)

    def run_train_generation(self,**kwargs):
        batch_size = self.train_cfg['batch_size']
        device = self.train_cfg['device']
        num_epoch = self.train_cfg['num_epoch']
        num_workers = self.train_cfg['num_workers']
        d_clip = self.train_cfg['d_clip']
        d_loss_thres = self.train_cfg['d_loss_thres']

        if 'gan_checkpoint_path' in kwargs.keys() and kwargs['gan_checkpoint_path'] is not None:
            gan_checkpoint_path = kwargs['gan_checkpoint_path']
        else:
            gan_checkpoint_path = self.train_cfg['checkpoints']['gan_checkpoints_path']

        if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
            log_path = kwargs['log_path']
        else:
            log_path = self.train_cfg['log']['log_path']

        log_step = self.train_cfg['log']['log_step']
        with_vis = self.train_cfg['log']['with_vis']
        vis_path = self.train_cfg['log']['vis_path']
        save_step=self.train_cfg['checkpoints']['save_step']
        self.print_key_args(gan_checkpoint_path=gan_checkpoint_path,
                            log_path=log_path,
                            train_data_path=self.train_cfg['train_data']['bg_data'],
                            device=device,with_matting=self.with_matting)

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, 'log.txt')
        log_fp = open(log_file, 'w')
        if with_vis:
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

        self.generator = self.generator.to(device)
        self.discrimator = self.discrimator.to(device)

        train_dataset = CloudMattingDataset(self.train_cfg['train_data']['bg_data'],
                                          self.train_cfg['train_data']['thickCloud_data'],
                                          self.train_cfg['train_data']['thinCloud_data'],
                                          self.train_cfg['train_data']['in_memory'])

        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers, drop_last=True,
                                                 pin_memory=True)
        train_dataiter = iter(train_dataloader)

        if not os.path.exists(gan_checkpoint_path):
            os.makedirs(gan_checkpoint_path)

        gan_criterion = builder_loss(batch_size=batch_size, device=device,
                                     **self.train_cfg['losses']['GANLoss'])
        generator_optimizer = build_optim(params=self.generator.parameters(), **self.train_cfg['generator_optimizer'])
        discrimator_optimizer = build_optim(params=self.discrimator.parameters(),
                                            **self.train_cfg['discrimator_optimizer'])

        state_dict, current_epoch, global_step = utils.load_model(gan_checkpoint_path)
        if state_dict is not None:
            result_keys = [x for x in state_dict.keys() if 'matting' in x]
            for k in result_keys:
                del state_dict[k]
            print('resume gan from epoch %d global_step %d' % (current_epoch, global_step))
            log_fp.writelines('resume gan model from epoch %d global_step %d\n' % (current_epoch, global_step))
            self.load_state_dict(state_dict, strict=False)

        summary = SummaryWriter(log_path)
        mm = 1
        self.train()
        self.generator.train()
        self.discrimator.train()
        for epoch in range(current_epoch, num_epoch):
            global_step += 1
            try:
                bg_img, thick_img, cloud_img = train_dataiter.__next__()
            except:
                train_dataiter = iter(train_dataloader)
                bg_img, thick_img, cloud_img = train_dataiter.__next__()

            bg_img = bg_img.to(device)
            thick_img = thick_img.to(device)
            cloud_img = cloud_img.to(device)

            generated_reflectance, generated_alpha, generated_img, fake_logits, real_logits = \
                self.forward(bg_img, thick_img, cloud_img)
            generator_loss, discrimator_loss = gan_criterion.forward(real_logits, fake_logits
                                                                     ,generated_reflectance)
            discrimator_optimizer.zero_grad()
            discrimator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discrimator.parameters(), d_clip)
            discrimator_optimizer.step()

            if torch.abs(discrimator_loss) < d_loss_thres:
                mm = mm + 5
            else:
                mm = 1
            for _ in range(mm):
                try:
                    bg_img, thick_img, cloud_img = train_dataiter.__next__()
                except:
                    train_dataiter = iter(train_dataloader)
                    bg_img, thick_img, cloud_img = train_dataiter.__next__()
                bg_img = bg_img.to(device)
                thick_img = thick_img.to(device)
                cloud_img = cloud_img.to(device)

                generated_reflectance, generated_alpha, generated_img, fake_logits, real_logits = \
                    self.forward(bg_img, thick_img, cloud_img)
                generator_loss, discrimator_loss = gan_criterion.forward(real_logits, fake_logits
                                                                         , generated_reflectance)

                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()
                if global_step % log_step == 1:
                    if with_vis:
                        utils.vis_cloud_matting(generated_img, generated_alpha, generated_reflectance,
                                                vis_path, '%d_label' % global_step)
                    print('[Epoch %d/%d] [generator loss:%f,discrimator loss:%f]' %
                          (epoch, num_epoch, generator_loss.item(),
                           discrimator_loss.item()))
                    log_fp.writelines('[Epoch %d/%d] [generator loss:%f,discrimator loss:%f]\n' %
                                      (epoch, num_epoch, generator_loss.item(),
                                       discrimator_loss.item()))
                    log_fp.flush()
                    summary.add_scalar('train/ganerator_loss', generator_loss, global_step)
                    summary.add_scalar('train/discrimator_loss', discrimator_loss, global_step)

            if global_step % save_step == 1:
                print('save model')
                utils.save_model(self, gan_checkpoint_path,
                                 epoch, global_step)
        print('save model')
        utils.save_model(self, gan_checkpoint_path,
                         epoch, global_step)

    def run_train_matting(self,**kwargs):
        batch_size = self.train_cfg['batch_size']
        device = self.train_cfg['device']
        num_epoch = self.train_cfg['num_epoch']
        num_workers = self.train_cfg['num_workers']
        save_step=self.train_cfg['checkpoints']['save_step']

        d_clip=self.train_cfg['d_clip']
        d_loss_thres = self.train_cfg['d_loss_thres']

        if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
            checkpoint_path=kwargs['checkpoint_path']
        else:
            checkpoint_path = self.train_cfg['checkpoints']['checkpoints_path']

        if 'gan_checkpoint_path' in kwargs.keys() and kwargs['gan_checkpoint_path'] is not None:
            gan_checkpoint_path=kwargs['gan_checkpoint_path']
        else:
            gan_checkpoint_path = self.train_cfg['checkpoints']['gan_checkpoints_path']

        if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
            log_path = kwargs['log_path']
        else:
            log_path = self.train_cfg['log']['log_path']

        log_step = self.train_cfg['log']['log_step']
        with_vis = self.train_cfg['log']['with_vis']
        vis_path = self.train_cfg['log']['vis_path']

        # self.print_key_args(checkpoint_path=checkpoint_path,
        #                     with_imagenet=self.backbone_cfg['pretrained'],
        #                     log_path=log_path,
        #                     train_data_path=self.train_cfg['train_data']['data_path'],
        #                     device=device)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, 'log.txt')
        log_fp = open(log_file, 'w')
        if with_vis:
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

        self.generator=self.generator.to(device)
        self.discrimator=self.discrimator.to(device)


        train_dataset = CloudMattingDataset(self.train_cfg['train_data']['bg_data'],self.train_cfg['train_data']['thickCloud_data'],
                                          self.train_cfg['train_data']['thinCloud_data'],
                                          self.train_cfg['train_data']['in_memory'])

        train_dataloader=data_utils.DataLoader(train_dataset,batch_size=batch_size,
                                               shuffle=True,num_workers=num_workers,drop_last=True,
                                               pin_memory=True)
        train_dataiter=iter(train_dataloader)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        gan_criterion = builder_loss(batch_size=batch_size,device=device,
                                     **self.train_cfg['losses']['GANLoss'])
        generator_optimizer = build_optim(params=self.generator.parameters(), **self.train_cfg['generator_optimizer'])
        discrimator_optimizer = build_optim(params=self.discrimator.parameters(), **self.train_cfg['discrimator_optimizer'])


        self.matting_model=self.matting_model.to(device)
        reflectance_criterion=builder_loss(**self.train_cfg['losses']['ReflectanceLoss'])
        alpha_criterion=builder_loss(**self.train_cfg['losses']['AlphaLoss'])
        factor=self.train_cfg['losses']['factor']
        matting_optimizer=build_optim(params=self.matting_model.parameters(),**self.train_cfg['matting_optimizer'])

        state_dict, current_epoch, global_step = utils.load_model(checkpoint_path)
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            log_fp.writelines('resume matting model from epoch %d global_step %d\n' % (current_epoch, global_step))
            self.load_state_dict(state_dict, strict=False)
        else:
            state_dict, current_epoch, global_step = utils.load_model(gan_checkpoint_path)
            if state_dict is not None:
                result_keys=[x for x in state_dict.keys() if 'matting' in x ]
                for k in result_keys:
                    del state_dict[k]
                print('resume gan from epoch %d global_step %d' % (current_epoch, global_step))
                log_fp.writelines('resume gan model from epoch %d global_step %d\n' % (current_epoch, global_step))
                self.load_state_dict(state_dict, strict=False)
        if 'lr_schedule' in self.train_cfg.keys():
            lr_schedule = build_lr_schedule(optimizer=matting_optimizer,
                                            **self.train_cfg['lr_schedule'])

        summary = SummaryWriter(log_path)
        time_metric=TimeMetric()

        self.train()
        self.generator.train()
        self.discrimator.train()

        for epoch in range(current_epoch, num_epoch):
            self.matting_model.train()
            global_step += 1
            try:
                bg_img, thick_img, cloud_img = train_dataiter.__next__()
            except:
                train_dataiter = iter(train_dataloader)
                bg_img, thick_img, cloud_img = train_dataiter.__next__()

            bg_img = bg_img.to(device)
            thick_img = thick_img.to(device)
            cloud_img = cloud_img.to(device)

            generated_reflectance,generated_alpha, generated_img, fake_logits, real_logits=\
                                    self.forward(bg_img,thick_img,cloud_img)
            generator_loss,discrimator_loss = gan_criterion.forward(real_logits,fake_logits
                                                            ,generated_reflectance)
            discrimator_optimizer.zero_grad()
            discrimator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discrimator.parameters(), d_clip)
            discrimator_optimizer.step()

            if torch.abs(discrimator_loss) < d_loss_thres:
                # mm = mm + 5
                mm=1
            else:
                mm = 1
            for _ in range(mm):
                try:
                    bg_img, thick_img, cloud_img = train_dataiter.__next__()
                except:
                    train_dataiter = iter(train_dataloader)
                    bg_img, thick_img, cloud_img = train_dataiter.__next__()
                bg_img = bg_img.to(device)
                thick_img = thick_img.to(device)
                cloud_img = cloud_img.to(device)

                generated_reflectance,generated_alpha, generated_img, fake_logits, real_logits = \
                    self.forward(bg_img, thick_img, cloud_img)
                generator_loss, discrimator_loss = gan_criterion.forward(real_logits, fake_logits
                                                                 ,generated_reflectance)

                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()



            #matting
            with torch.no_grad():
                generated_reflectance, generated_alpha, generated_img, fake_logits, real_logits = \
                    self.forward(bg_img, thick_img, cloud_img)

            generated_img=self.matting_model.resize_img(generated_img.detach())
            generated_reflectance=self.matting_model.resize_img(generated_reflectance.detach())
            generated_alpha = self.matting_model.resize_img(generated_alpha.detach())

            pred_alpha,pred_reflectance=self.matting_model.forward(generated_img)

            reflectance_loss=reflectance_criterion(pred_reflectance,generated_reflectance)
            alpha_loss=alpha_criterion(pred_alpha,generated_alpha)

            total_loss=reflectance_loss*factor[1]+alpha_loss*factor[2]

            matting_optimizer.zero_grad()
            total_loss.backward()
            matting_optimizer.step()
            if 'lr_schedule' in self.train_cfg.keys():
                lr_schedule.step()
                summary.add_scalar('learning rate',
                                   matting_optimizer.state_dict()['param_groups'][0]['lr'],global_step)
            if global_step % log_step==1:
                fps = time_metric.get_fps(log_step * batch_size)
                time_metric.reset()
                if with_vis:
                    utils.vis_cloud_matting(generated_img, generated_alpha, generated_reflectance,
                                            vis_path, '%d_label' % global_step)
                print('[Epoch %d/%d] [generator loss:%f,discrimator loss:%f]' %
                      (epoch, num_epoch, generator_loss.item(),
                       discrimator_loss.item()))
                log_fp.writelines('[Epoch %d/%d] [generator loss:%f,discrimator loss:%f]\n' %
                                  (epoch, num_epoch, generator_loss.item(),
                                   discrimator_loss.item()))
                log_fp.flush()
                summary.add_scalar('train/ganerator_loss', generator_loss, global_step)
                summary.add_scalar('train/discrimator_loss', discrimator_loss, global_step)
                print('train stage: [Epoch %d/%d] [reflectance loss:%f,alpha loss:%f,'
                      'total_loss:%f] [fps:%f]' %
                      (epoch, num_epoch, reflectance_loss.item(),
                       alpha_loss.item(),
                       total_loss.item(), fps))
                log_fp.writelines('train stage: [Epoch %d/%d] [reflectance loss:%f,alpha loss:%f'
                                  ',total_loss:%f] [fps:%f]\n' %
                                  (epoch, num_epoch, reflectance_loss.item(), alpha_loss.item(),
                                   total_loss.item(), fps))
                log_fp.flush()
                summary.add_scalar('train/reflectance_loss', reflectance_loss, global_step)
                summary.add_scalar('train/alpha_loss', alpha_loss, global_step)
                summary.add_scalar('train/total_loss', total_loss, global_step)


                if with_vis:
                    utils.vis_cloud_matting(generated_img, pred_alpha, pred_reflectance,
                                            vis_path, '%d_pred'%global_step,trimap=None)


            if global_step % save_step==1:
                utils.save_model(self, checkpoint_path,
                                     current_epoch, global_step,prefix='cloud_matting')


    def run_cloud_generation(self,**kwargs):
        batch_size = self.test_cfg['batch_size']
        device = self.train_cfg['device']

        num_workers = self.train_cfg['num_workers']

        if 'gan_checkpoint_path' in kwargs.keys() and kwargs['gan_checkpoint_path'] is not None:
            gan_checkpoint_path = kwargs['gan_checkpoint_path']
        else:
            gan_checkpoint_path = self.train_cfg['checkpoints']['gan_checkpoints_path']

        if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
            log_path = kwargs['log_path']
        else:
            log_path = self.train_cfg['log']['log_path']


        with_vis = self.train_cfg['log']['with_vis']
        vis_path = self.train_cfg['log']['vis_path']

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(log_path, 'log.txt')
        log_fp = open(log_file, 'w')
        if with_vis:
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
        self.generator = self.generator.to(device)
        self.discrimator = self.discrimator.to(device)

        train_dataset = CloudMattingDataset(self.train_cfg['train_data']['bg_data'],
                                          self.train_cfg['train_data']['thickCloud_data'],
                                          self.train_cfg['train_data']['thinCloud_data'],
                                          self.train_cfg['train_data']['in_memory'])

        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers, drop_last=False,
                                                 pin_memory=True)

        if not os.path.exists(gan_checkpoint_path):
            os.makedirs(gan_checkpoint_path)
        state_dict, current_epoch, global_step = utils.load_model(gan_checkpoint_path)
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            log_fp.writelines('resume matting model from epoch %d global_step %d\n' % (current_epoch, global_step))
            self.load_state_dict(state_dict, strict=False)


        self.generator.eval()
        self.discrimator.eval()
        for data in tqdm.tqdm(train_dataloader):

            global_step += 1

            bg_img, thick_img, cloud_img = data
            bg_img = bg_img.to(device)
            thick_img = thick_img.to(device)
            cloud_img = cloud_img.to(device)


            generated_reflectance, generated_alpha, generated_img, fake_logits, real_logits = \
                self.forward(bg_img, thick_img, cloud_img)

            utils.vis_cloud_generation(generated_img,generated_alpha,generated_reflectance,
                                       vis_path,'%d_genenration'%global_step)

    def run_test_interface(self,**kwargs):
        mode=self.test_cfg['mode']
        if mode=='matting':
            self.run_test_matting(**kwargs)
        else:
            raise NotImplementedError('mode: %s is not available'% mode)

    def run_test_matting(self,**kwargs):
        batch_size = self.test_cfg['batch_size']
        device = self.test_cfg['device']
        num_workers = self.test_cfg['num_workers']
        if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
            checkpoint_path = kwargs['checkpoint_path']
        else:
            checkpoint_path = self.test_cfg['checkpoints']['checkpoints_path']

        if 'data_path' in kwargs.keys() and kwargs['data_path'] is not None:
            self.test_cfg['test_data']['data_path'] = kwargs['data_path']

        if 'data_format' in kwargs.keys() and kwargs['data_format'] is not None:
            self.test_cfg['test_data']['data_format'] = kwargs['data_format']

        with_vis = self.test_cfg['log']['with_vis']
        if 'result_path' in kwargs.keys() and kwargs['result_path'] is not None:
            vis_path=kwargs['result_path']
        else:
            vis_path = self.test_cfg['log']['vis_path']

        self.print_key_args(checkpoint_path=checkpoint_path,
                            with_vis=with_vis,
                            vis_path=vis_path,
                            data_path=self.test_cfg['test_data']['data_path'],
                            device=device)

        if with_vis:
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

        self.generator = self.generator.to(device)
        self.discrimator = self.discrimator.to(device)
        self.matting_model.to(device)

        test_dataset = CloudSingleDataset(**self.test_cfg['test_data'])
        test_dataloader = data_utils.DataLoader(test_dataset, batch_size,
                                                 shuffle=False, drop_last=False,
                                                 num_workers=num_workers,pin_memory=True)

        state_dict, current_epoch, global_step = utils.load_model(checkpoint_path)
        if state_dict is not None:
            print('resume from epoch %d global_step %d' % (current_epoch, global_step))
            self.load_state_dict(state_dict, strict=True)
        else:
            raise NotImplementedError("%s does not exsits;"%checkpoint_path)

        time_metric = TimeMetric()
        for data in tqdm.tqdm(test_dataloader):
            self.eval()
            self.matting_model.eval()
            cloud_img, img_files = data
            cloud_img = cloud_img.to(device)

            pred_alpha,pred_reflectance= \
                self.matting_model.forward(cloud_img)
            if with_vis:

                utils.vis_test_cloud_matting(cloud_img,pred_alpha,pred_reflectance,img_files,vis_path,
                                             trimap=None)
        total_ms=time_metric.get_time_ms()
        print('total time:{0}'.format(total_ms))


