import wandb
from .dataloader import SICEGradTrain,SICEGradTest,SICEGradVal, SICEMixTest, LOLTrain, get_training_augmentation, get_validation_augmentation, get_transform
import torch
from tqdm import tqdm as tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from .models.lptn_model import LPTNModel
from torchsummary import summary




def eval(root_dir, dset, kernel_loss_weight, lr,loss_weight = 2000,gan_type = 'standard', use_hypernet = True, device='cuda', nrb_top = 4, nrb_high = 5, nrb_low = 3):

    transform = get_transform(dataset='grad')
    
    if(dset=='mix'):
        test_dataset = SICEMixTest(root_dir=root_dir, augmentation= get_training_augmentation(), transform= transform)

    elif(dset=='grad'):
        test_dataset = SICEGradTest(root_dir=root_dir, augmentation= get_training_augmentation(), transform= transform)
    
    elif(dset=='lol'):
        test_dataset = subdirectories = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, name))]
        high = subdirectories[1]
        low = subdirectories[0]
        print(high, low)
        test_dataset = LOLTrain(high_res_folder=high, low_res_folder=low, flag=2, augmentation=get_training_augmentation())

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    lptn_model = LPTNModel(loss_weight, kernel_loss_weight, device, lr, gan_type=gan_type, use_hypernet=use_hypernet, nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top)
    summary(lptn_model.net_g , input_size=(3, 608, 896))
    total_loss = []
    psnr_test,ssim_test, lpips_test = 0,0,0
    psnr_test,ssim_test, lpips_test = 0,0,0

    with tqdm(
        test_loader
    ) as loader:
        for iteration,batch_data in enumerate(loader):
            x,y = batch_data
            lptn_model.net_g.eval()
            lptn_model.feed_data(x,y)
            lptn_model.optimize_parameters(iteration)
            break
        
    psnr_test_max = [0,0]
    flag = 0

    with tqdm(
        test_loader
    ) as loader:
        if(os.path.exists('./best_model_g.pth')):
            lptn_model.load_network('./best_model_g.pth', device=device)
        else:
            lptn_model.load_network('/kaggle/input/besthypernettruemodel/best_model_g.pth', device=device)
        for iteration,batch_data in enumerate(loader):
            x,y = batch_data
            
            lptn_model.net_g.eval()
            
            lptn_model.feed_data(x,y)
            
            loss_iter,kernel_los_iter,psnr_test_iter,ssim_test_iter, lpips_test_iter = lptn_model.optimize_parameters(iteration)
            lptn_model.visualise(iteration=iteration)
            flag = 0
            
            lpips_test += lpips_test_iter
            psnr_test += psnr_test_iter
            ssim_test += ssim_test_iter
            total_loss.append(loss_iter)
    
    lpips_test /=(iteration+1)
    psnr_test /= (iteration+1)
    ssim_test /= (iteration+1)
    lpips_test /= (iteration+1)
    avg_loss = sum(total_loss)/len(total_loss)

    print(f'TEST LPIPS {lpips_test}')
    print(f'TEST PSNR {psnr_test}')
    print(f'TEST SSIM {ssim_test}')


def eval_model(configs):
    eval(configs['root_dir'],
        configs['dset'],
        configs['kernel_loss_weight'], 
        configs['lr'],
        configs['loss_weight'],
        configs['gan_type'],
        configs['use_hypernet'],
        configs['device'],
        configs['nrb_top'],
        configs['nrb_high'],
        configs['nrb_low']
        )