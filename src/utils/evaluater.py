import wandb
from .dataloader import SICETestDataset,SICEGradTest,SICEMixTest
import torch
from tqdm import tqdm as tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from .models.lptn_model import LPTNModel
# from torchsummary import summary




def eval(root_dir, lr,loss_weight = 2000,gan_type = 'standard' ,device='cuda', nrb_top = 4, nrb_high = 5, nrb_low = 3,exposure='over',path='/kaggle/working/best_model_g.pth'):

    # testing_indices = [
    #     *range(4, 24), 28, 31, 33, 34, 
    #     *range(37, 40), *range(46, 53), 
    #     *range(55, 70), *range(75, 80), 
    #     *range(100, 104)
    # ]

    # # Initialize the test dataset
    # test_dataset = SICETestDataset(
    #     root_dir=r'/kaggle/input/sicedataset',
    #     exposure_type='over',
    #     indices=testing_indices
    # )

    # # Create the DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type, nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top,levels=[0,1,2],weights=[0.5,0.3,0.2])
    # # summary(lptn_model.net_g , input_size=(3, 608, 896))
    # total_loss = []
    # psnr_test,ssim_test, lpips_test,mssim_test = 0,0,0,0

 
    # with tqdm(
    #     test_loader
    # ) as loader:

    #     lptn_model.load_network(path, device=device)
    #     for iteration,batch_data in enumerate(loader):
    #         x,y = batch_data
            
    #         lptn_model.net_g.eval()
            
    #         lptn_model.feed_data(x,y)
            
    #         psnr_test_iter,ssim_test_iter, lpips_test_iter,mssim_iter_test = lptn_model.optimize_parameters(iteration,mode='test')
    #         # lptn_model.visualise(iteration=iteration)
    #         flag = 0
            
    #         lpips_test += lpips_test_iter
    #         psnr_test += psnr_test_iter
    #         ssim_test += ssim_test_iter
    #         mssim_test+=mssim_iter_test
    # lpips_test /=(iteration+1)
    # psnr_test /= (iteration+1)
    # ssim_test /= (iteration+1)
    # mssim_test /= (iteration+1)

    # print(f'TEST LPIPS over {lpips_test}')
    # print(f'TEST PSNR over  {psnr_test}')
    # print(f'TEST SSIM over {ssim_test}')
    # print(f'TEST MSSIM over {mssim_test}')

    # testing_indices = [
    #     *range(4, 24), 28, 31, 33, 34, 
    #     *range(37, 40), *range(46, 53), 
    #     *range(55, 70), *range(75, 80), 
    #     *range(100, 104)
    # ]

    # # # Initialize the test dataset
    # test_dataset = SICETestDataset(
    #     root_dir=r'/kaggle/input/sicedataset',
    #     exposure_type='under',
    #     indices=testing_indices
    # )

    # # Create the DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type, nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top,levels=[0,1,2],weights=[0.5,0.3,0.2])
    # # summary(lptn_model.net_g , input_size=(3, 608, 896))
    # total_loss = []
    # psnr_test,ssim_test, lpips_test,mssim_test = 0,0,0,0

 
    # with tqdm(
    #     test_loader
    # ) as loader:

    #     lptn_model.load_network(path, device=device)
    #     for iteration,batch_data in enumerate(loader):
    #         x,y = batch_data
            
    #         lptn_model.net_g.eval()
            
    #         lptn_model.feed_data(x,y)
            
    #         psnr_test_iter,ssim_test_iter, lpips_test_iter,mssim_iter_test = lptn_model.optimize_parameters(iteration,mode='test')
    #         # lptn_model.visualise(iteration=iteration)
    #         flag = 0
            
    #         lpips_test += lpips_test_iter
    #         psnr_test += psnr_test_iter
    #         ssim_test += ssim_test_iter
    #         mssim_test+=mssim_iter_test
    # lpips_test /=(iteration+1)
    # psnr_test /= (iteration+1)
    # ssim_test /= (iteration+1)
    # mssim_test /= (iteration+1)

    # print(f'TEST LPIPS under {lpips_test}')
    # print(f'TEST PSNR under  {psnr_test}')
    # print(f'TEST SSIM under {ssim_test}')
    # print(f'TEST MSSIM under {mssim_test}')

    # # # Initialize the test dataset
    test_dataset = SICEMixTest(
        root_dir=r'/kaggle/input/sice-grad-and-sice-mix/SICEGM',
    )

    # Create the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type, nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top,levels=[0,1,2],weights=[0.5,0.3,0.2])
    # summary(lptn_model.net_g , input_size=(3, 608, 896))
    total_loss = []
    psnr_test,ssim_test, lpips_test,mssim_test = 0,0,0,0

 
    with tqdm(
        test_loader
    ) as loader:

        lptn_model.load_network(path, device=device)
        for iteration,batch_data in enumerate(loader):
            x,y = batch_data
            
            lptn_model.net_g.eval()
            
            lptn_model.feed_data(x,y)
            
            psnr_test_iter,ssim_test_iter, lpips_test_iter,mssim_iter_test = lptn_model.optimize_parameters(iteration,mode='test')
            lptn_model.visualise(iteration=iteration)
            flag = 0
            
            lpips_test += lpips_test_iter
            psnr_test += psnr_test_iter
            ssim_test += ssim_test_iter
            mssim_test+=mssim_iter_test
    lpips_test /=(iteration+1)
    psnr_test /= (iteration+1)
    ssim_test /= (iteration+1)
    mssim_test /= (iteration+1)

    print(f'TEST LPIPS mix {lpips_test}')
    print(f'TEST PSNR mix  {psnr_test}')
    print(f'TEST SSIM mix {ssim_test}')
    print(f'TEST MSSIM mix {mssim_test}')

    # test_dataset = SICEGradTest(
    #     root_dir=r'/kaggle/input/sice-grad-and-sice-mix/SICEGM',
    # )

    # # Create the DataLoader
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type, nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top,levels=[0,1,2],weights=[0.5,0.3,0.2])
    # # summary(lptn_model.net_g , input_size=(3, 608, 896))
    # total_loss = []
    # psnr_test,ssim_test, lpips_test,mssim_test = 0,0,0,0

 
    # with tqdm(
    #     test_loader
    # ) as loader:

    #     lptn_model.load_network(path, device=device)
    #     for iteration,batch_data in enumerate(loader):
    #         x,y = batch_data
            
    #         lptn_model.net_g.eval()
            
    #         lptn_model.feed_data(x,y)
            
    #         psnr_test_iter,ssim_test_iter, lpips_test_iter,mssim_iter_test = lptn_model.optimize_parameters(iteration,mode='test')
    #         # lptn_model.visualise(iteration=iteration)
    #         flag = 0
            
    #         lpips_test += lpips_test_iter
    #         psnr_test += psnr_test_iter
    #         ssim_test += ssim_test_iter
    #         mssim_test+=mssim_iter_test
    # lpips_test /=(iteration+1)
    # psnr_test /= (iteration+1)
    # ssim_test /= (iteration+1)
    # mssim_test /= (iteration+1)

    # print(f'TEST LPIPS grad {lpips_test}')
    # print(f'TEST PSNR grad  {psnr_test}')
    # print(f'TEST SSIM grad {ssim_test}')
    # print(f'TEST MSSIM grad {mssim_test}')

def eval_model(configs):
    eval(configs['root_dir'],
        configs['lr'],
        configs['loss_weight'],
        configs['gan_type'],
        configs['device'],
        configs['nrb_top'],
        configs['nrb_high'],
        configs['nrb_low'],
        configs['exposure'],
        configs['model_path']
        )