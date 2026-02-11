import math
import torch
from torch import nn
import cv2
from thop import profile, clever_format
from . import loss
import torchvision.transforms.functional as F
from .cldice import soft_dice_cldice 

import kornia as K
import kornia.filters as KF

def unscale(img, min_max=(0, 1)):
    img = (img-min_max[0])/(min_max[1] - min_max[0]) 
    return img


class CssGenerator(nn.Module):
    def __init__(
        self, segment_fn,
        image_size,
        channels=3,
        loss_type='l1',

        opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.segment_fn = segment_fn
        self.loss_type = loss_type
        self.opt=opt

        self.kernel_size = (21, 21)  # approx for sigma=5.0; use odd sizes
        self.sigma = (5.0, 5.0)

#############defining loss functions###########
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        self.loss_nce = []
        for i in range(3):
            self.loss_nce.append(loss.MaskPatchNCELoss().to(device))
        
        self.loss_gan = loss.GANLoss('lsgan').to(device)
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_cldice=soft_dice_cldice().to(device)
        self.loss_bce=nn.BCELoss().to(device)
        



    def p_sample_segment(self, x_in, opt):
        x_start_ = x_in['A']
        segm_V = torch.zeros_like(x_start_)
        dsize = x_start_.shape[-1]

        if opt['phase'] != 'train':
            if 'STARE' in opt['datasets']['test']['dataroot'] or 'DRIVE' in opt['datasets']['test']['dataroot'] or '30XCA' in opt['datasets']['test']['dataroot']:
                for opt1 in range(0,dsize,256):
                    for opt2 in range(0,dsize,256):
                        x_start = x_start_[:, :, opt1:opt1+256, opt2:opt2+256]
                        segm_V[:, :, opt1:opt1+256, opt2:opt2+256] = self.segment_fn(torch.cat([x_start, x_start * 2.0 + KF.gaussian_blur2d(x_start, self.kernel_size, self.sigma) * (-1.0)], dim=1))
                return segm_V
            
        for opt1 in range(2):
            for opt2 in range(2):
                x_start = x_start_[:, :, opt1::2, opt2::2]
                segm_V[:, :, opt1::2, opt2::2] = self.segment_fn(torch.cat([x_start, x_start * 2.0 + KF.gaussian_blur2d(x_start, self.kernel_size, self.sigma) * (-1.0)], dim=1))
        return segm_V


    @torch.no_grad()
    def segment(self, x_in, opt):
        return self.p_sample_segment(x_in, opt)


########## this function has been written to calculate the model parameters and flops for the segmentation network. It is called from the main training loop during inference. ##########

    @torch.no_grad()
    def findprofile(self, x_in, opt):
        x_in['A']=x_in['A'].unsqueeze(0)
        print(f'Calculating FLOPS and Params for segmentation network...')
        flops, params = profile(self.segment_fn, inputs=(torch.cat([x_in['A'], x_in['A']], dim=2,)), verbose=False)
        return flops, params


    def p_losses(self, x_in, noise=None):
        a_start = x_in['A']
        unsharp_image = a_start * 2.0 + KF.gaussian_blur2d(a_start, self.kernel_size, self.sigma) * (-1.0)
        device = a_start.device
        [b, c, h, w] = a_start.shape


        #### Segmentation phase ####
        mask_V = self.segment_fn(torch.cat([a_start, unsharp_image], dim=1))
        
        #### Image transfer phase ####
        fractal = torch.eye(2,device=device)[:, torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)].transpose(0, 1)
        synt_A = self.segment_fn(torch.cat([ a_start,unsharp_image], dim=1), fractal.to(device)) # the transferred image

        #### Cycle path () ####
        unsharp_image2 = synt_A * 2.0 + KF.gaussian_blur2d(synt_A, self.kernel_size, self.sigma) * (-1.0)
        mask_F = self.segment_fn(torch.cat([synt_A, synt_A], dim=1))
        mask_V1 = torch.eye(2,device=device)[:, torch.clamp_min(mask_V[:, 0], 0).type(torch.long)].transpose(0, 1)
        l_recon=self.loss_func(self.segment_fn(torch.cat([a_start, unsharp_image], dim=1),mask_V1),a_start) 
        l_recon=l_recon.sum() / int(b * c * h * w)


        new_X=unscale(x_in['F'],min_max=(-1,1))
        new_X=(new_X > 0.6).float() * 1

        l_l1=self.loss_l1(mask_F,x_in['F'])
        l_cldice=self.loss_cldice(unscale(mask_F,min_max=(-1,1)),new_X)
        l_bce=self.loss_bce(unscale(mask_F,min_max=(-1,1)),new_X)

        return [ x_in['A'], x_in['F'], mask_V, synt_A, mask_F], [l_recon, l_l1, l_bce, l_cldice] #
        
    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
    


