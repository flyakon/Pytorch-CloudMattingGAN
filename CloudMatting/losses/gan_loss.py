import torch
import torch.nn as nn

class VanillaLoss():

    def __init__(self,penalty_factor=1.,batch_size=2,device='cuda:0',**kwargs):
        self.criterion=nn.BCEWithLogitsLoss()
        self.penalty_factor=penalty_factor
        self.batch_size=batch_size
        self.device=device
        self.fake_labels=torch.zeros([self.batch_size,1],device=self.device)
        self.real_labels = torch.ones([self.batch_size,1],  device=self.device)

    def forward(self,D_logits_real, D_logits_fake,G_relectance):
        max_rgb_value,_ = G_relectance.max(dim=1)
        min_rgb_value,_ = G_relectance.min(dim=1)
        S_value = (max_rgb_value - min_rgb_value) / (max_rgb_value + 1e-3)
        S_penalty = self.penalty_factor * torch.norm(S_value,p=2)
        # penalty = torch.norm(generated_pi, p=2)
        penalty = 0.
        # penalty=torch.max(penalty,torch.tensor(0.,device=self.device))*self.penalty_factor
        D_loss_real=self.criterion(D_logits_real,self.real_labels)
        D_loss_fake=self.criterion(D_logits_fake,self.fake_labels)
        D_loss = D_loss_real + D_loss_fake

        G_loss = self.criterion(D_logits_fake,
                                    self.real_labels) + penalty+S_penalty
        return G_loss,D_loss


class LSGANLoss():
    def __init__(self,penalty_factor=1.,**kwargs):
        self.penalty_factor=penalty_factor

    def forward(self,D_logits_real, D_logits_fake,G_relectance):
        max_rgb_value, _ = G_relectance.max(dim=1)
        min_rgb_value, _ = G_relectance.min(dim=1)
        S_value = (max_rgb_value - min_rgb_value) / (max_rgb_value + 1e-3)
        S_penalty = self.penalty_factor * torch.norm(S_value, p=2)
        # penalty = torch.norm(generated_pi, p=2)
        penalty=0.
        D_loss = 0.5 * (torch.mean((D_logits_real - 1) ** 2) + torch.mean(D_logits_fake ** 2))
        G_loss = 0.5 * torch.mean((D_logits_fake - 1) ** 2) +S_penalty + penalty
        return G_loss,D_loss


class WGANLosss():

    def __init__(self,penalty_factor=1.,**kwargs):
        self.penalty_factor=penalty_factor


    def forward(self,D_logits_real, D_logits_fake,G_relectance):
        max_rgb_value, _ = G_relectance.max(dim=1)
        min_rgb_value, _ = G_relectance.min(dim=1)
        S_value = (max_rgb_value - min_rgb_value) / (max_rgb_value + 1e-3)
        S_penalty = self.penalty_factor * torch.norm(S_value, p=2)
        # penalty = torch.norm(generated_pi, p=2)
        penalty=0.
        D_loss = - (torch.mean(D_logits_real) - torch.mean(D_logits_fake))
        G_loss = - torch.mean(D_logits_fake) + penalty + S_penalty
        return G_loss,D_loss

