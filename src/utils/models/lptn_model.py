import importlib
import torch
from collections import OrderedDict
from os import path as osp
import matplotlib.pyplot as plt
from utils.models.base_model import BaseModel
from utils.models.losses import compute_gradient_penalty


from .archs.LPTN_paper_arch import LPTNPaper
from .archs.discriminator_arch import Discriminator
from .losses.losses import MSELoss, GANLoss

loss_module = importlib.import_module('utils.models.losses')

class LPTNModel(BaseModel):

    def __init__(self, loss_weight, device, lr, gan_type='standard', nrb_low=3, nrb_high=5, nrb_top=4):
        super(LPTNModel, self).__init__(loss_weight, device, lr)

        self.gan_type = gan_type
        self.nrb_low = nrb_low
        self.nrb_high = nrb_high
        self.nrb_top = nrb_top

        # creating discriminator object
        self.device = torch.device(device)
        disc = Discriminator()
        disc = disc.to(self.device)

        # creating model object
        model = LPTNPaper(
        nrb_low=self.nrb_low,
        nrb_high=self.nrb_high,
        num_high=2,
        device = self.device,
        nrb_top=self.nrb_top
        )
        
        # using model as generator
        self.net_g = model.to(self.device)
        self.print_network(self.net_g)

        self.net_d = disc.to(self.device)
        self.print_network(self.net_d)

        self.init_training_settings()

        # initialize losses
        self.MLoss = MSELoss(loss_weight=self.loss_weight, reduction='mean').to(self.device)
        self.GLoss = GANLoss(gan_type=self.gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1).to(self.device)
    
    def load_network(self, load_path,device = 'cuda', strict=True):
        
        load_net = torch.load(load_path, map_location=device)
        self.net_g.load_state_dict(load_net, strict=strict)
        print("Network is loaded")      

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
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
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                 lr=0.0001, weight_decay=0, betas=[0.9, 0.99])                     

        self.optimizers.append(self.optimizer_d)

    def feed_data(self, RGB, TLR, THR):
        """
        Args:
            RGB: RGB image
            TLR: Thermal Low Resolution
            THR: Thermal High Resoution
        """
        self.RGB = RGB.to(self.device)
        self.Thermal_low_res = TLR.to(self.device)
        self.Thermal_high_res = THR.to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.RGB, self.Thermal_low_res)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            # pixel loss
            l_g_pix = self.MLoss(self.output, self.Thermal_high_res).to(self.device)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.GLoss(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        self.output = self.net_g(self.RGB, self.Thermal_low_res)
        
        # real
        real_d_pred = self.net_d(self.Thermal_high_res)
        l_d_real = self.GLoss(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        # fake
        fake_d_pred = self.net_d(self.output)
        l_d_fake = self.GLoss(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        gradient_penalty = compute_gradient_penalty(self.net_d, self.Thermal_high_res, self.output, self.device)
        l_d = l_d_real + l_d_fake + self.gp_weight * gradient_penalty

        l_d.backward()
        self.optimizer_d.step()
        
        visuals = self.get_current_visuals()
        input_img = visuals['RGB'] 
        result_img = visuals['result']
        if 'Thermal_high_res' in visuals:
            THR_img = visuals['Thermal_high_res']
            del self.Thermal_high_res
      
        psnr_t,ssim_t = self.calculate_metrics(result_img, THR_img)       
            
        return l_g_total, psnr_t, ssim_t

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.RGB, self.Thermal_low_res)
        self.net_g.train()

    def nondist_validation(self, dataloader):
        psnr = 0
        ssim = 0

        for idx,batch in enumerate(dataloader):
            rgb,thermal_low_res,thermal_high_res = batch
            self.feed_data(rgb,thermal_low_res,thermal_high_res)
            self.test()
            
            visuals = self.get_current_visuals()
            input_img = visuals['RGB'] 
            result_img = visuals['result']
            if 'Thermal_high_res' in visuals:
                THR_img = visuals['Thermal_high_res']
                del self.Thermal_high_res

            x, y = self.calculate_metrics(result_img,THR_img) 
            psnr = x + psnr
            ssim = y + ssim  

        psnr /= (idx + 1)
        ssim /= (idx + 1)
                
        return psnr, ssim
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Thermal_low_res'] = self.Thermal_low_res.detach().to(self.device)
        out_dict['result'] = self.output.detach().to(self.device)
        out_dict['RGB'] = self.RGB.detach().to(self.device)
        if hasattr(self, 'Thermal_high_res'):
            out_dict['Thermal_high_res'] = self.Thermal_high_res.detach().to(self.device)
        return out_dict

    def save(self, path):
        self.save_network(self.net_g, 'net_g', path+'_g.pth')
        self.save_network(self.net_d, 'net_d', path+'_d.pth')
