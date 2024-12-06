import wandb
from .dataloader import Dataset, Dataset_flir, Dataset_cats, Dataset_UAV, create_list2,sort_files_by_name
from .transformations import get_training_augmentation, get_training_augmentation_flir, get_validation_augmentation_flir, resize, resize_cats, get_training_augmentation_cats, get_validation_augmentation_cats
import torch
from tqdm import tqdm as tqdm
import yaml
from torch.utils.data import DataLoader
from .models.lptn_model import LPTNModel

import pandas as pd

def train(epochs,
          batch_size,
          rgb_dir,
          tar_dir,
          th_low_res_dir,
          rgb_val_dir,
          tar_val_dir,
          th_low_res_val_dir,
          rgb_test_dir,
          tar_test_dir,
          th_low_res_test_dir,
          rgb_dir_flir,
          tar_dir_flir,
        #   seed,
          device='cuda',
          lr=1e-4,
          loss_weight = 2000,
          dataset = 'CATS',
          gan_type = 'standard',
          nrb_low = 3,
          nrb_high = 5,
          cats_directory = '/kaggle/input/cats-dataset',
          rgb_uav_dir = 'VGTSR/HR RGB/',
          thermal_low_res_uav_dir = 'VGTSR/LR thermal/4/BI/',
          thermal_high_res_uav_dir = 'VGTSR/GT thermal/',
          nrb_top = 4
          ):

    torch.manual_seed(seed=3)
        
    if dataset == 'ULB':
        
        train_dataset = Dataset(
        rgb_dir,
        th_low_res_dir,
        tar_dir,
        augmentation=get_training_augmentation(), 
        preprocessing= True)
        
        valid_dataset = Dataset(
        rgb_val_dir,
        th_low_res_val_dir,
        tar_val_dir,
        augmentation=None, 
        preprocessing= True)
        
        test_dataset = Dataset(
        rgb_test_dir,
        th_low_res_test_dir,
        tar_test_dir,
        augmentation=None, 
        preprocessing= True)
        
    elif dataset == 'CATS':
        
        total_color,total_thermal = create_list2(cats_directory)
        
        train_dataset = Dataset_cats(
            total_color[:250],
            total_thermal[:250],
            augmentation=get_training_augmentation_cats(), 
            preprocessing= True,
            resize = resize_cats())
        
        valid_dataset = Dataset_cats(
            total_color[250:300],
            total_thermal[250:300],
            augmentation=get_validation_augmentation_cats(),
            preprocessing= True,
            resize = resize_cats())
        
        test_dataset = Dataset_cats(
            total_color[300:],
            total_thermal[300:],
            augmentation=get_validation_augmentation_cats(), 
            preprocessing= True,
            resize = resize_cats())
        
    elif dataset == 'UAV':
        
        thermal_high_res_sorted, thermal_low_res_sorted, rgb_sorted = sort_files_by_name(thermal_high_res_uav_dir,thermal_low_res_uav_dir,rgb_uav_dir)
        
        train_dataset = Dataset_UAV(
        rgb_sorted[:800],
        thermal_low_res_sorted[:800],
        thermal_high_res_sorted[:800],
        augmentation=get_training_augmentation(), 
        preprocessing= True)
        
        valid_dataset = Dataset_UAV(
        rgb_sorted[800:900],
        thermal_low_res_sorted[800:900],
        thermal_high_res_sorted[800:900],
        augmentation=None, 
        preprocessing= True)
        
        test_dataset = Dataset_UAV(
        rgb_sorted[900:],
        thermal_low_res_sorted[900:],
        thermal_high_res_sorted[900:],
        augmentation=None, 
        preprocessing= True)
        
        
    else:
        
        train_data_flir = pd.read_csv("./Dataset/FLIR/train_data.csv")
        val_data_flir = pd.read_csv("./Dataset/FLIR/val_data.csv")
        test_data_flir = pd.read_csv("./Dataset/FLIR/test_data.csv")
        
        train_dataset = Dataset_flir(
        rgb_dir_flir,
        tar_dir_flir,
        train_data_flir.iloc[:,0],
        train_data_flir.iloc[:,1],
        augmentation=get_training_augmentation_flir(), 
        preprocessing= True,
        resize = resize())
        
        valid_dataset = Dataset_flir(
        rgb_dir_flir,
        tar_dir_flir,
        val_data_flir.iloc[:,0],
        val_data_flir.iloc[:,1],
        augmentation=get_validation_augmentation_flir(), 
        preprocessing= True,
        resize = resize())
        
        test_dataset = Dataset_flir(
        rgb_dir_flir,
        tar_dir_flir,
        test_data_flir.iloc[:,0],
        test_data_flir.iloc[:,1],
        augmentation=get_validation_augmentation_flir(), 
        preprocessing= True,
        resize = resize())
        

    lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type, nrb_low=nrb_low, nrb_high=nrb_high, nrb_top=nrb_top)

    def worker_seed(worker_id):
        torch.manual_seed(3 + worker_id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=worker_seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)#, drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle = True)

    # print(dataset)
    a,b,c = train_dataset.__getitem__(0)
    # print("rgb",a.shape)
    # print("thermal low res",b.shape)
    # print("thermal high res",c.shape)
    max_ssim = 0
    max_psnr = 0
    logger = {'epoch': 0,'train_loss': 0, 'train_psnr': 0, 'train_ssim': 0, 'val_ssim': 0, 'val_psnr': 0, 'test_ssim': 0, 'test_psnr': 0}
    for i in range(0, epochs):
        total_loss = []
        psnr_train,ssim_train = 0,0

        with tqdm(
            train_loader,
            desc = "Training Progress"
        ) as loader:
            for iteration,batch_data in enumerate(loader):
                x,y,z = batch_data
                # lptn_model.update_learning_rate(iteration)
                lptn_model.feed_data(x,y,z)
                loss_iter,psnr_train_iter,ssim_train_iter = lptn_model.optimize_parameters(iteration)
                total_loss.append(loss_iter)
                psnr_train = psnr_train + psnr_train_iter
                ssim_train = ssim_train + ssim_train_iter
                
        psnr_train /= (iteration+1)
        ssim_train /= (iteration+1)
        avg_loss = sum(total_loss)/len(total_loss)
    
        psnr_val, ssim_val = lptn_model.nondist_validation(valid_loader)

        logger['train_loss'] = avg_loss
        logger['train_psnr'] = psnr_train
        logger['train_ssim'] = ssim_train
        logger['val_psnr'] = psnr_val
        logger['val_ssim'] = ssim_val
        logger['epoch'] = i
        if max_ssim <= logger['val_ssim']:
            max_ssim = logger['val_ssim']
            max_psnr = logger['val_psnr'] 
            wandb.config.update({'max_ssim':max_ssim,'max_psnr':max_psnr,'best_epoch':i}, allow_val_change=True)
            lptn_model.save('./best_model')
            
        wandb.log(logger)
    lptn_model.load_network('./best_model_g.pth', device=device)
    psnr_test,ssim_test = lptn_model.nondist_validation(test_loader)
    print('test_ssim:',ssim_test,'test_psnr:',psnr_test)
    wandb.config.update({'test_ssim':ssim_test,'test_psnr':psnr_test}, allow_val_change=True)  


def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['rgb_dir'],
         configs['tar_dir'], configs['th_low_res_dir'], configs['rgb_val_dir'],
         configs['tar_val_dir'], configs['th_low_res_val_dir'],configs['rgb_test_dir'],
         configs['tar_test_dir'], configs['th_low_res_test_dir'], #configs['seed'],
         configs['rgb_dir1'],configs['tar_dir1'],
         configs['device'], configs['lr'],
         configs['loss_weight'], configs['dataset'], configs['gan_type'], 
         configs['nrb_low'], configs['nrb_high'],configs['cats_directory'], configs['rgb_uav_dir'],
         configs['thermal_low_res_uav_dir'], 
         configs['thermal_high_res_uav_dir'],
         configs['nrb_top']
         )
         
