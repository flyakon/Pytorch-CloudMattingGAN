'''
@anthor: Wenyuan Li
@desc: Networks for Cloud matting
@date: 2020/8/5
'''
import torch
import torchvision
import numpy as np
import torch.nn as nn
from ..base_net import BaseNet
from ..backbone.builder import build_backbone
import torch.utils.data as data_utils
import tqdm


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.conv = nn.Sequential(*[
            nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True),
        ])

        self.downsample = nn.Sequential(*[
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=self.stride, padding=0),
            nn.BatchNorm2d(self.out_channel),
        ])

    def forward(self, x):
        out = self.conv(x)
        idenfity = self.downsample(x)
        out = out + idenfity
        return out


class CloudMattingNet(BaseNet):
    def __init__(self,
                 backbone_cfg: dict,
                 neck_cfg:dict,
                 head_cfg: dict,
                 train_cfg: dict,
                 test_cfg: dict, **kwargs):
        super(CloudMattingNet, self).__init__()
        self.backbone = build_backbone(**backbone_cfg)
        self.build_arch(head_cfg)
        self.build_neck(neck_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.img_size = head_cfg['img_size']
        self.backbone_cfg = backbone_cfg

    def build_neck(self,neck_cfg:dict):
        in_channels = neck_cfg['in_channels']
        out_channels = neck_cfg['out_channels']
        self.conv1=nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)



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

        self.trans_conv6 = nn.Sequential(*[
            nn.Conv2d(feat_channels[4], feat_channels[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[5]),
            nn.ReLU(inplace=True)])

        self.pred_reflectance = nn.Conv2d(feat_channels[5], 3, kernel_size=3, stride=1, padding=1)
        self.pred_alpha = nn.Conv2d(feat_channels[5], 3, kernel_size=3, stride=1, padding=1)


    def resize_img(self, x):
        x = torch.nn.functional.interpolate(x, (self.img_size, self.img_size))
        return x


    def forward(self, x, **kwargs):
        input_data = x
        x, endpoints = self.backbone(x)
        x=self.conv1(x)
        endpoints['block6'] = x
        x=self.pool1(x)
        x=self.conv2(x)

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv1(x)
        if 'block6' in endpoints.keys():
            x = x + endpoints['block6']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv2(x)
        if 'block5' in endpoints.keys():
            x = x + endpoints['block5']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv3(x)
        if 'block4' in endpoints.keys():
            x = x + endpoints['block4']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv4(x)
        if 'block3' in endpoints.keys():
            x = x + endpoints['block3']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv5(x)
        if 'block2' in endpoints.keys():
            x = x + endpoints['block2']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2, mode='bilinear')
        x = self.trans_conv6(x)
        if 'block1' in endpoints.keys():
            x = x + endpoints['block1']

        alpha_logits = self.pred_alpha(x)
        reflectance_logits = self.pred_reflectance(x)

        return alpha_logits, reflectance_logits

    # def run_train_interface(self, **kwargs):
    #     batch_size = self.train_cfg['batch_size']
    #     device = self.train_cfg['device']
    #     num_epoch = self.train_cfg['num_epoch']
    #     num_workers = self.train_cfg['num_workers']
    #     if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
    #         checkpoint_path = kwargs['checkpoint_path']
    #     else:
    #         checkpoint_path = self.train_cfg['checkpoints']['checkpoints_path']
    #     if 'log_path' in kwargs.keys() and kwargs['log_path'] is not None:
    #         log_path = kwargs['log_path']
    #     else:
    #         log_path = self.train_cfg['log']['log_path']
    #
    #     save_step = self.train_cfg['checkpoints']['save_step']
    #     log_step = self.train_cfg['log']['log_step']
    #     with_vis = self.train_cfg['log']['with_vis']
    #     vis_path = self.train_cfg['log']['vis_path']
    #
    #     self.print_key_args(checkpoint_path=checkpoint_path,
    #                         with_imagenet=self.backbone_cfg['pretrained'],
    #                         log_path=log_path,
    #                         train_data_path=self.train_cfg['train_data']['data_path'],
    #                         device=device)
    #     if not os.path.exists(log_path):
    #         os.makedirs(log_path)
    #     log_file = os.path.join(log_path, 'log.txt')
    #     log_fp = open(log_file, 'w')
    #     if with_vis:
    #         if not os.path.exists(vis_path):
    #             os.makedirs(vis_path)
    #     self.to(device)
    #     train_dataset = CloudMattingDataset(**self.train_cfg['train_data'])
    #     train_dataloader = data_utils.DataLoader(train_dataset, batch_size,
    #                                              shuffle=True, drop_last=False,
    #                                              num_workers=num_workers)
    #     if not os.path.exists(checkpoint_path):
    #         os.makedirs(checkpoint_path)
    #
    #     alpha_criterion = builder_loss(**self.train_cfg['losses']['AlphaLoss'])
    #     reflectance_criterion = builder_loss(**self.train_cfg['losses']['ReflectanceLoss'])
    #     factors = self.train_cfg['losses']['factor']
    #     optimizer = build_optim(params=self.parameters(), **self.train_cfg['optimizer'])
    #     if 'lr_schedule' in self.train_cfg.keys():
    #         lr_schedule = build_lr_schedule(optimizer=optimizer, **self.train_cfg['lr_schedule'])
    #
    #     state_dict, current_epoch, global_step = utils.load_model(checkpoint_path)
    #     if state_dict is not None:
    #         print('resume from epoch %d global_step %d' % (current_epoch, global_step))
    #         log_fp.writelines('resume from epoch %d global_step %d' % (current_epoch, global_step))
    #         self.load_state_dict(state_dict, strict=True)
    #
    #     summary = SummaryWriter(log_path)
    #     time_metric = TimeMetric()
    #     for epoch in range(current_epoch, num_epoch):
    #         for i, data in enumerate(train_dataloader):
    #             global_step += 1
    #             self.train()
    #             cloud_img, alpha_label, reflectance_label = data
    #             cloud_img = cloud_img.to(device)
    #             alpha_label = alpha_label.to(device)
    #             reflectance_label = reflectance_label.to(device)
    #
    #             pred_alpha, pred_reflectance = self.forward(cloud_img)
    #             alpha_loss = alpha_criterion(pred_alpha, alpha_label)
    #             reflectance_loss = reflectance_criterion(pred_reflectance, reflectance_label)
    #             train_loss = alpha_loss * factors[0] + reflectance_loss * factors[1]
    #
    #             optimizer.zero_grad()
    #             train_loss.backward()
    #             optimizer.step()
    #
    #             if global_step % log_step == 1:
    #                 if with_vis:
    #                     utils.vis_cloud_matting(cloud_img, pred_alpha, pred_reflectance,
    #                                             vis_path, '%d_pred' % global_step)
    #                     utils.vis_cloud_matting(cloud_img, alpha_label, reflectance_label,
    #                                             vis_path, '%d_label' % global_step)
    #                 fps = time_metric.get_fps(log_step * batch_size)
    #                 time_metric.reset()
    #                 print('[Epoch %d/%d] [Batch %d/%d] [alpha loss:%f,reflectance loss:%f, total loss: %f] [fps:%f]' %
    #                       (epoch, num_epoch, i, len(train_dataloader), alpha_loss.item(),
    #                        reflectance_loss.item(), train_loss.item(), fps))
    #                 log_fp.writelines(
    #                     '[Epoch %d/%d] [Batch %d/%d] [alpha loss:%f,reflectance loss:%f, total loss: %f] [fps:%f]\n' %
    #                     (epoch, num_epoch, i, len(train_dataloader), alpha_loss.item(),
    #                      reflectance_loss.item(), train_loss.item(), fps))
    #                 log_fp.flush()
    #                 summary.add_scalar('train/alpha_loss', alpha_loss, global_step)
    #                 summary.add_scalar('train/reflectance_loss', reflectance_loss, global_step)
    #                 summary.add_scalar('train/total_loss', train_loss, global_step)
    #
    #         if 'lr_schedule' in self.train_cfg.keys():
    #             lr_schedule.step()
    #             summary.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'],
    #                                global_step)
    #         lr = optimizer.param_groups[0]['lr']
    #         summary.add_scalar('learning_rate', lr, global_step)
    #         if epoch % save_step == 1:
    #             print('save model')
    #             utils.save_model(self, checkpoint_path,
    #                              epoch, global_step, max_keep=200)
    #
    # def run_test_interface(self, **kwargs):
    #     batch_size = self.test_cfg['batch_size']
    #     device = self.test_cfg['device']
    #     num_workers = self.test_cfg['num_workers']
    #     if 'checkpoint_path' in kwargs.keys() and kwargs['checkpoint_path'] is not None:
    #         checkpoint_path = kwargs['checkpoint_path']
    #     else:
    #         checkpoint_path = self.test_cfg['checkpoints']['checkpoints_path']
    #
    #     with_vis = self.test_cfg['log']['with_vis']
    #     vis_path = self.test_cfg['log']['vis_path']
    #
    #     self.print_key_args(checkpoint_path=checkpoint_path,
    #                         with_vis=with_vis,
    #                         vis_path=vis_path,
    #                         data_path=self.test_cfg['test_data']['data_path'],
    #                         device=device)
    #
    #     if with_vis:
    #         if not os.path.exists(vis_path):
    #             os.makedirs(vis_path)
    #     self.to(device)
    #
    #     test_dataset = CloudMattingDataset(**self.test_cfg['test_data'])
    #     test_dataloader = data_utils.DataLoader(test_dataset, batch_size,
    #                                             shuffle=False, drop_last=False,
    #                                             num_workers=num_workers)
    #
    #     state_dict, current_epoch, global_step = utils.load_model(checkpoint_path)
    #     if state_dict is not None:
    #         print('resume from epoch %d global_step %d' % (current_epoch, global_step))
    #         self.load_state_dict(state_dict, strict=True)
    #     else:
    #         raise NotImplementedError("%s does not exsits;" % checkpoint_path)
    #
    #     time_metric = TimeMetric()
    #     for data in tqdm.tqdm(test_dataloader):
    #         self.eval()
    #         cloud_img, img_files = data
    #         cloud_img = cloud_img.to(device)
    #
    #         pred_alpha, pred_reflectance = self.forward(cloud_img)
    #         if with_vis:
    #             utils.vis_test_cloud_matting(cloud_img, pred_alpha, pred_reflectance, img_files, vis_path)
    #     total_ms = time_metric.get_time_ms()
    #     print('total time:{0}'.format(total_ms))


