
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np

from utils.models import lr_scheduler as lr_scheduler

class BaseModel():
    """Base model."""

    def __init__(self, loss_weight, device, lr):
        self.loss_weight = loss_weight
        self.device = device
        self.lr = lr
        self.schedulers = []
        self.optimizers = []
        self.is_train = True
        self.P = PeakSignalNoiseRatio().to(self.device)
        self.Z = StructuralSimilarityIndexMeasure().to(self.device)
        self.L = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean', normalize=True).to(self.device)

    def feed_data(self, LLI, HLI):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        self.nondist_validation(dataloader, current_iter)

    def validation_speed(self, dataloader, times_per_img=50, num_imgs=20, size=None):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        self.nondist_validation_speed(dataloader, times_per_img, num_imgs, size)

    def get_current_log(self):
        return self.log_dict

    def setup_schedulers(self):
        """Set up schedulers."""

        for optimizer in self.optimizers:
            self.schedulers.append(
                lr_scheduler.MultiStepRestartLR(optimizer, 
                                                milestones=[50000, 100000, 200000, 300000], 
                                                gamma=0.5))

    # @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        #net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [
            param_group['lr']
            for param_group in self.optimizers[0].param_groups
        ]

    def save_network(self, net, net_label,path):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """

        #net = self.get_bare_model(net)
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.to(self.device)
        torch.save(state_dict, path)
        
    def calculate_metrics(self, img1, img2):
            
            img1 = img1.clamp_(0, 1)
            img2 = img2.clamp_(0, 1)

            LPIP_iter = self.L(img1,img2).to(self.device)
            # img1 = img1
            # img1 = img1.round().int()
            # img1 = img1.float()

            # img2 = img2
            # # img2 = img2.round().int()
            # # img2 = img2.float()

            return self.P(img1, img2).to(self.device), self.Z(img1, img2).to(self.device), LPIP_iter        
