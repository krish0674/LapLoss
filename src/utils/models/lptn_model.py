import importlib
import torch
import cv2
import numpy as np
from collections import OrderedDict
from os import path as osp
import os
import matplotlib.pyplot as plt
from utils.models.base_model import BaseModel
from utils.models.losses import compute_gradient_penalty

from torch import nn
from .archs.LPTN_paper_arch import LPTNPaper
from .archs.lptn import LPTN

from .archs.LPTN_paper_arch import Lap_Pyramid_Conv
from .archs.discriminator_arch import Discriminator1,Discriminator2,Discriminator3
from .losses.losses import MSELoss, GANLoss

loss_module = importlib.import_module('utils.models.losses')

class LPTNModel(BaseModel):

    def __init__(self, loss_weight, device, lr, gan_type='standard', nrb_low=3, nrb_high=5, nrb_top=4,levels=[0,1,2],weights=[4/7,2/7,1/7]):
        super(LPTNModel, self).__init__(loss_weight, device, lr)

        self.gan_type = gan_type
        self.nrb_low = nrb_low
        self.nrb_high = nrb_high
        self.nrb_top = nrb_top
        self.num_high = 2  
        self.levels=levels
        self.weights=weights
        #define multiple here 
        # creating discriminator object
        self.device = torch.device(device)

        disc1 = Discriminator1()
        disc1 = disc1.to(self.device)
        disc2 = Discriminator2()
        disc2 = disc2.to(self.device)
        disc3 = Discriminator3()
        disc3 = disc3.to(self.device)

        # creating model object
        model = LPTNPaper(
        nrb_low =self.nrb_low,
        nrb_high =self.nrb_high,
        nrb_top =self.nrb_top,
        num_high= self.num_high,
        device=self.device,
        )

        # Instantiate Lap_Pyramid_Conv
        self.lap_pyramid = Lap_Pyramid_Conv(
            num_high= self.num_high,
            device=self.device,
        )
        
        # using model as generator
        self.net_g = model.to(self.device)
        self.print_network(self.net_g)

        self.net_d1 = disc1.to(self.device)
        self.net_d2 = disc2.to(self.device)
        self.net_d3 = disc3.to(self.device)

        self.print_network(self.net_d1)
        self.print_network(self.net_d2)
        self.print_network(self.net_d3)


        self.init_training_settings()
        
        glw = 1
        print("GAN TURNED OFF" if glw==0 else "GAN TURNED ON")

        self.MLoss = MSELoss(loss_weight=self.loss_weight, reduction='mean').to(self.device)
        self.GLoss = GANLoss(gan_type=self.gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=glw).to(self.device)
        
        #optimal kernel
        self.opt_kernel = torch.tensor([[1., 4., 6., 4., 1],
                                            [4., 16., 24., 16., 4.],
                                            [6., 24., 36., 24., 6.],
                                            [4., 16., 24., 16., 4.],
                                            [1., 4., 6., 4., 1.]])
        self.opt_kernel /= 256.
        self.opt_kernel = self.opt_kernel.to(device)
    
    def load_network(self, load_path,device = 'cuda', strict=True):
        
        load_net = torch.load(load_path, map_location=device)
        self.net_g.load_state_dict(load_net, strict=strict)
        print("Network is loaded")      

    def init_training_settings(self):
        self.net_g.train()
        self.net_d1.train()
        self.net_d2.train()
        self.net_d3.train()
        self.optimizers = []
        self.gp_weight = 100
        self.net_d_iters = 1
        self.net_d_init_iters = 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
    
        self.optimizer_g = torch.optim.Adam(optim_params,
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        self.optimizer_d1 = torch.optim.Adam(self.net_d1.parameters(),
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     

        self.optimizers.append(self.optimizer_d1)

        self.optimizer_d2 = torch.optim.Adam(self.net_d2.parameters(),
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     

        self.optimizers.append(self.optimizer_d2)

        self.optimizer_d3 = torch.optim.Adam(self.net_d3.parameters(),
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     

        self.optimizers.append(self.optimizer_d3)

    # def pyramid_decom(self, img):
    #     current = img
    #     pyr = []
    #     for _ in range(self.num_high):
            
    #         filtered = self.lap_pyramid.conv_gauss(current, kernel)
    #         down = self.lap_pyramid.downsample(filtered)
    #         up = self.lap_pyramid.upsample(down, kernel)
    #         if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
    #             up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode)
    #         diff = current - up
    #         pyr.append(diff)
    #         current = down
    #     pyr.append(current)
    #     return pyr
    
    def feed_data(self, LLI, HLI):
        """
        Args:
            LLI : Low Light Image
            HLI : High Light Image
        """
        self.LLI = LLI.to(self.device)
        self.HLI = HLI.to(self.device)
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                                        [4., 16., 24., 16., 4.],
                                        [6., 24., 36., 24., 6.],
                                        [4., 16., 24., 16., 4.],
                                        [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(3, 1, 1, 1)

        self.pyr_gt=self.lap_pyramid.pyramid_decom(self.HLI)

    def calculate_weighted_loss(self, pyr_gt, pyr_pred, discriminators, levels, weights):
        """
        Calculate the weighted loss at specified pyramid levels with multiple discriminators.
        Args:
            pyr_gt: Ground truth pyramid levels (list of tensors).
            pyr_pred: Predicted pyramid levels (list of tensors).
            discriminators: List of discriminator models, one for each pyramid level.
            levels: List of pyramid levels to consider (e.g., [0, 1, 2]).
            weights: List of weights corresponding to the specified levels.
        Returns:
            total_loss: Weighted total loss.
            loss_dict: Dictionary of individual losses at each level.
        """
        total_loss = 0.0
        loss_dict = {}

        # Iterate only through the specified levels
        for level, weight in zip(levels, weights):
            if level >= len(pyr_gt) or level >= len(pyr_pred) or level >= len(discriminators):
                raise ValueError(f"Specified level {level} exceeds available pyramid levels.")
            
            # Get the ground truth, predicted, and discriminator for the current level
            gt = pyr_gt[level]
            pred = pyr_pred[level]
            discriminator = discriminators[level]
            pred = (pred - pred.mean()) / (pred.std() + 1e-8)
            gt = (gt - gt.mean()) / (gt.std() + 1e-8)            # print(f"at level {level} shape is {gt.shape}")
            # Pixel loss at this level
            l_pix = self.MLoss(pred, gt).to(self.device)

            # GAN loss at this level
            fake_g_pred = discriminator(pred)
            l_gan = self.GLoss(fake_g_pred, True, is_disc=False)

            # Weighted loss

            level_loss = weight * (l_pix + l_gan)
            # print(f"At level {level}, mse loss si {l_pix}")
            # print(f"At level {level},gan loss is {l_gan}")
            # print(f"At level {level},loss si {level_loss}")
            total_loss += level_loss

            # Store individual level loss
            # loss_dict[f'level_{level}'] = level_loss.item()
        # print(f"Total loss is {total_loss}")
        return total_loss #, loss_dict


    def optimize_parameters(self, current_iter,mode='train'):
        torch.autograd.set_detect_anomaly(True)

        # optimize net_g
        for p in self.net_d1.parameters():
            p.requires_grad = False
        for p in self.net_d2.parameters():
            p.requires_grad = False
        for p in self.net_d3.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        _,self.output = self.net_g(self.LLI)
        pyr_pred=self.lap_pyramid.pyramid_decom(self.output)
        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            # pixel loss
            discriminators = [self.net_d1, self.net_d2, self.net_d3]
            l_g_total = self.calculate_weighted_loss(self.pyr_gt, pyr_pred, discriminators,self.levels,self.weights)
            # Backpropagation and optimization
            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d1.parameters():
            p.requires_grad = True
        for p in self.net_d2.parameters():
            p.requires_grad = True
        for p in self.net_d3.parameters():
            p.requires_grad = True

        self.optimizer_d1.zero_grad()
        self.optimizer_d2.zero_grad()
        self.optimizer_d3.zero_grad()

        # List of discriminators, their optimizers, and pyramid levels
        discriminators = [self.net_d1, self.net_d2, self.net_d3]
        optimizers = [self.optimizer_d1, self.optimizer_d2, self.optimizer_d3]
        pyr_gt_levels = self.pyr_gt
        pyr_pred_levels = pyr_pred

        # Loop through each discriminator
        for i, (discriminator, optimizer, pyr_gt, pyr_pred) in enumerate(zip(discriminators, optimizers, pyr_gt_levels, pyr_pred_levels)):
            pyr_gt = pyr_gt.detach()
            pyr_pred = pyr_pred.detach()
            pyr_gt = (pyr_gt - pyr_gt.mean()) / (pyr_gt.std() + 1e-8)
            pyr_pred = (pyr_pred - pyr_pred.mean()) / (pyr_pred.std() + 1e-8)  
            # Real
            real_d_pred = discriminator(pyr_gt)
            l_d_real = self.GLoss(real_d_pred, True, is_disc=True)

            # Fake
            fake_d_pred = discriminator(pyr_pred)
            l_d_fake = self.GLoss(fake_d_pred, False, is_disc=True)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, pyr_gt, pyr_pred, self.device)
            l_d = l_d_real + l_d_fake + self.gp_weight * gradient_penalty

            # Backpropagation and optimization
            l_d.backward()
            optimizer.step()

        
        visuals = self.get_current_visuals()
        input_img = visuals['Low_Limage'] 
        result_img = visuals['result']
        if 'High_Limage' in visuals:
            HLI_img = visuals['High_Limage']
            #del self.HLI

        if mode=='train':
            psnr_t,ssim_t = self.calculate_metrics(result_img,HLI_img)

            return l_g_total, psnr_t, ssim_t
        if mode=='test':
            psnr_t,ssim_t,lpips,mssim = self.calculate_metrics_test(result_img,HLI_img)

            return psnr_t,ssim_t,lpips,mssim
    

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            _,self.output = self.net_g(self.LLI)
            pyr_pred=self.lap_pyramid.pyramid_decom(self.output)

        self.net_g.train()

    def nondist_validation(self, dataloader):
        psnr = 0
        ssim = 0
        lpips = 0

        for idx,batch in enumerate(dataloader):
            low_limage, high_limage = batch
            self.feed_data(low_limage, high_limage)
            self.test()
            
            visuals = self.get_current_visuals()
            input_img = visuals['Low_Limage'] 
            result_img = visuals['result']
            if 'High_Limage' in visuals:
                HLI_img = visuals['High_Limage']
                del self.HLI

            x, y= self.calculate_metrics(result_img,HLI_img)
            psnr = x + psnr
            ssim = y + ssim
            # lpips = z + lpips


        # print(psnr)
        # print(ssim)
        psnr /= (idx + 1)
        ssim /= (idx + 1)
        # lpips /= (idx + 1)
        
        print(f'Val PSNR {psnr}')
        print(f'Val SSIM {ssim}')
                
        return psnr, ssim
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Low_Limage'] = self.LLI.detach().to(self.device)
        out_dict['result'] = self.output.detach().to(self.device)
        if hasattr(self, 'HLI'):
            out_dict['High_Limage'] = self.HLI.detach().to(self.device)
        return out_dict

    def save(self, path):
        self.save_network(self.net_g, 'net_g', path+'_g.pth')
        self.save_network(self.net_d1, 'net_d1', path+'_d.pth')
        self.save_network(self.net_d2, 'net_d2', path+'_d.pth')
        self.save_network(self.net_d3, 'net_d3', path+'_d.pth')
        
    def visualise(self, save_dir='output_images', iteration=0):
        _,output = self.net_g(self.LLI)
        # print(self.LLI)
        # print(self.LLI.shape)
        # print(self.HLI.shape)
        # print(output.shape)
        input = self.LLI
        label = self.HLI
        
        os.makedirs(save_dir, exist_ok=True)
        
        unique_index = iteration

        label = label.detach().cpu().numpy()
        label = np.transpose(label, (0, 2, 3, 1))  # CHW to HWC
        label = label[0]

        label = (label * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'label_image_{unique_index}.png'), label)

        output = output.detach().cpu().numpy()
        output = np.transpose(output, (0, 2, 3, 1))  # CHW to HWC
        #print(output.shape)
        img = output[0]
        img = (img * 255.).astype(np.uint8)  # Scale to [0, 255]

        mean = [0.41441402, 0.41269127, 0.37940571]
        std = [0.33492465, 0.33443474, 0.33518072]

        input = input.detach().cpu().numpy()
        input = np.transpose(input, (0, 2, 3, 1))  # CHW to HWC
        img_in = input[0]
        #img_in = (img_in*std)+mean
        img_in = (img_in * 255.).astype(np.uint8)

        #print(img.shape)
        #print(img_in.shape)
        print("imaged")
        cv2.imwrite(os.path.join(save_dir, f'output_image_{unique_index}.png'), img)
        cv2.imwrite(os.path.join(save_dir, f'input_image_{unique_index}.png'), img_in)
