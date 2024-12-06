import torch
import torch.nn as nn
import torch.nn.functional as F

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, device=torch.device('cuda')):
        super(Lap_Pyramid_Conv, self).__init__()
        self.interpolate_mode = 'bicubic'
        self.num_high = num_high
        self.device = device
        self.kernel = nn.Parameter(torch.zeros(1, 1, 5, 5), requires_grad=True)  # Placeholder for the kernel
        
    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x, kernel):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, kernel)

    def conv_gauss(self, img, kernel):
        device = torch.device('cuda')
        img = img.to(device)
        kernel = kernel.to(device)
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out
    
    def pyramid_decom(self, img, kernel):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, kernel)
            down = self.downsample(filtered)
            up = self.upsample(down, kernel)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image

class EnetHyper(nn.Module):
    def __init__(self, ray_hidden_dim=256):
        super().__init__()
        
        self.image_mlp = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(68096, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, 25)
        )

    def forward(self, image):
        kernel = self.image_mlp(image)
        kernel = kernel.view(image.shape[0], 1, 5, 5)
        return kernel

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks):
        super(Trans_low, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)   
        
    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out
    
class Trans_top(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_top, self).__init__()
        
        self.num_high = num_high
        
        model = [nn.Conv2d(3, 64, 3, padding=1),
             nn.LeakyReLU()]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]
        
        model += [nn.Conv2d(64, 3, 3, padding=1)]
        
        # code from trans_mask_block - 1x1 convolution
        model += [nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, up_mask):
        topped_mask = self.model(up_mask)
        return topped_mask

class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(Trans_high, self).__init__()
    
        self.num_high = num_high
        
        model = [nn.Conv2d(9, 64, 3, padding=1),
             nn.LeakyReLU()]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]
        
        model += [nn.Conv2d(64, 3, 3, padding=1)]
        
        # code from trans_mask_block - 1x1 convolution
        model += [nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, high_with_low):
        mask = self.model(high_with_low)
        return mask

class LPTNPaper(nn.Module):
    def __init__(self, nrb_low=5, nrb_high=3, nrb_top=3, num_high=3, use_hypernet =False, device=torch.device('cuda')):
        super(LPTNPaper, self).__init__()
        
        self.use_hypernet = use_hypernet
        self.device = device
        self.interpolate_mode = 'bicubic'
        
        self.lap_pyramid = Lap_Pyramid_Conv(num_high,self.device).to(self.device)
        trans_low = Trans_low(nrb_low)
        trans_high = Trans_high(num_residual_blocks=nrb_high, num_high=num_high)
        trans_top = Trans_top(num_residual_blocks=nrb_top, num_high=num_high)
        self.trans_low = trans_low.to(self.device)
        self.trans_high = trans_high.to(self.device)
        self.trans_top = trans_top.to(self.device)
        if(self.use_hypernet):
            self.hyper_net = EnetHyper().to(self.device)

    def forward(self, real_A_full):
        if self.use_hypernet == True:
            kernel = self.hyper_net(real_A_full)
            kernel = kernel[0].repeat(3, 1, 1, 1)
        else:
            kernel = torch.tensor([[1., 4., 6., 4., 1],
                                            [4., 16., 24., 16., 4.],
                                            [6., 24., 36., 24., 6.],
                                            [4., 16., 24., 16., 4.],
                                            [1., 4., 6., 4., 1.]])
            kernel /= 256.
            kernel = kernel.repeat(3, 1, 1, 1)

        # Update the kernel parameter in lap_pyramid
        self.lap_pyramid.kernel.data = kernel
        
        # initial laplacian pyramid
        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full, kernel=kernel)
        
        # upsampling lowest pyramid level
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        
        # translating lowest pyramid level
        fake_B_low = self.trans_low(pyr_A[-1])
        
        # upsampling translation of the lowest pyramid level
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
            
        # concatenation of 2nd last image, upsampled last image, upsampled and translated last image
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        
        # output of trans_high
        mask = self.trans_high(high_with_low)
        
        # upsampled mask
        mask_up = nn.functional.interpolate(mask, size=(pyr_A[-3].shape[2], pyr_A[-3].shape[3]), mode=self.interpolate_mode, align_corners=True)
        
        # upsampled mask passed through trans_top
        topped_mask = self.trans_top(mask_up)
        
        # product of trans_top's o/p and pyr[-3] (OG)
        result_top = torch.mul(topped_mask, pyr_A[-3]).to(self.device)
        
        # product of trans_high's o/p and pyr[-2] (OG)
        result_high = torch.mul(mask, pyr_A[-2]).to(self.device)
        
        pyr_A_trans = []
        pyr_A_trans.append(result_top)
        pyr_A_trans.append(result_high)
        pyr_A_trans.append(fake_B_low)
        
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        
        return fake_B_full