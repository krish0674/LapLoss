import wandb
from .dataloader import SICEGradTrain,SICEGradTest,SICEGradVal, SICEMixTrain,SICEMixVal, LOLTrain, get_training_augmentation, get_validation_augmentation, get_transform
import torch
from tqdm import tqdm as tqdm
import albumentations as A
import os
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from .models.lptn_model import LPTNModel

import pandas as pd

def train(epochs,
          batch_size,
          dset,
          root_dir,
          nrb_top = 4,
          nrb_high = 5,
          nrb_low = 3,
          device='cuda',
          lr=1e-4,
          loss_weight = 2000,
          gan_type = 'standard',
          ):
    
    transform = get_transform(dataset='grad')
    if(dset=='mix'):
        train_dataset = SICEMixTrain(root_dir=root_dir, augmentation= get_training_augmentation())
        val_dataset = SICEMixVal(root_dir = root_dir, augmentation= get_validation_augmentation())

    elif(dset=='grad'):
        train_dataset = SICEGradTrain(root_dir=root_dir, augmentation= get_training_augmentation())
        val_dataset = SICEGradVal(root_dir = root_dir, augmentation= get_validation_augmentation())

    elif(dset=='lol'):
        subdirectories = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, name))]
        high = subdirectories[1]
        low = subdirectories[0]
        print(high, low)
        train_dataset = LOLTrain(high_res_folder=high, low_res_folder=low, flag=0, augmentation=get_training_augmentation())
        val_dataset = LOLTrain(high_res_folder=high, low_res_folder=low, flag=1, augmentation=get_training_augmentation())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type, nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top)

    # a,b = train_dataset.__getitem__(0)
    # print("LLI image",a.shape)
    # print("HLI image",b.shape)
    max_ssim = 0
    max_psnr = 0
    logger = {'epoch': 0,'train_loss': 0, 'train_psnr': 0, 'train_ssim': 0,'train_lpips': 0, 'val_ssim': 0, 'val_psnr': 0, 'val_lpips' : 0, 'test_ssim': 0, 'test_psnr': 0}    
    for i in range(0, epochs):
        total_loss = []
        kernel_loss =[]
        psnr_train,ssim_train = 0,0

        with tqdm(
            train_loader,
            desc = "Training Progress"
        ) as loader:
            for iteration,batch_data in enumerate(loader):
                x,y = batch_data
                # lptn_model.update_learning_rate(iteration)
                lptn_model.feed_data(x,y)
                # print(f'Iteration {iteration}, input min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
                # print(f'Iteration {iteration}, target min: {y.min()}, max: {y.max()}, mean: {y.mean()}')

                loss_iter,psnr_train_iter,ssim_train_iter, lpips_train_iter = lptn_model.optimize_parameters(iteration)
               #print(f'Iteration {iteration}, loss: {loss_iter}, PSNR: {psnr_train_iter}, SSIM: {ssim_train_iter}')

                total_loss.append(loss_iter)
                psnr_train = psnr_train + psnr_train_iter
                ssim_train = ssim_train + ssim_train_iter
                lpips_train = lpips_train = lpips_train_iter
                
    
        psnr_train /= (iteration+1)
        ssim_train /= (iteration+1)
        lpips_train /= (iteration+1)
        avg_loss = sum(total_loss)/len(total_loss)
        
        print(f'TRAIN PSNR {psnr_train}')
        print(f'TRAIN SSIM {ssim_train}')
        print(f'TRAIN LPIPS {lpips_train}')
    
        psnr_val, ssim_val, lpips_val = lptn_model.nondist_validation(valid_loader)

        logger['train_loss'] = avg_loss
        logger['train_psnr'] = psnr_train
        logger['train_ssim'] = ssim_train
        logger['val_psnr'] = psnr_val
        logger['val_lpips'] = lpips_val
        logger['val_ssim'] = ssim_val
        logger['epoch'] = i
        if max_ssim <= logger['val_ssim']:
            max_ssim = logger['val_ssim']
            max_psnr = logger['val_psnr'] 
            wandb.config.update({'max_ssim':max_ssim,'max_psnr':max_psnr,'best_epoch':i}, allow_val_change=True)
            lptn_model.save('./best_model')
            
        wandb.log(logger)
        #lptn_model.load_network('./best_model_g.pth', device=device)
        
        # if i == 2:
        #    lptn_model.load_network(sf_path, device=device) 
    #psnr_test,ssim_test = lptn_model.nondist_validation(valid_loader)
    #print('test_ssim:',ssim_test,'test_psnr:',psnr_test)
    # wandb.config.update({'test_ssim':ssim_test,'test_psnr':psnr_test}, allow_val_change=True)
    
    #lptn_model.load_network(sf_path, device=device)
    lptn_model.load_network('./best_model_g.pth', device=device)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    with tqdm(
        val_loader
    ) as loader:
        for iteration,batch_data in enumerate(loader):
            x,y = batch_data
            lptn_model.net_g.eval()
            lptn_model.feed_data(x,y)
            lptn_model.visualise()
            print('visualized')
            break


def train_model(configs):
    train(configs['epochs'], 
        configs['batch_size'],
        configs['dset'],
        configs['root_dir'],
        configs['nrb_top'],
        configs['nrb_high'],
        configs['nrb_low'],
        configs['device'], 
        configs['lr'],
        configs['loss_weight'],
        configs['gan_type'],
        )