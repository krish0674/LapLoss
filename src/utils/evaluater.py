from .dataloader import SICETestDataset, SICEGradTest, SICEMixTest, SICEAllImagesTestDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .models.lptn_model import LPTNModel


def build_test_dataset(dset, root_dir, exposure, tf):
    if dset == 'sice':
        # all exposures of a single scene folder from Dataset_Part1
        return SICEAllImagesTestDataset(root_dir=root_dir, test_folder_id=tf)
    if dset == 'grad':
        # root_dir must contain SICE_Grad/ and SICE_Reshape/
        return SICEGradTest(root_dir=root_dir)
    if dset == 'mix':
        # root_dir must contain SICE_Mix/ and SICE_Reshape/
        return SICEMixTest(root_dir=root_dir)
    raise ValueError(f"Unknown dset '{dset}'. Choose from 'sice', 'grad', 'mix'.")


def eval(root_dir, lr, loss_weight=2000, gan_type='standard', device='cuda',
         nrb_top=3, nrb_high=4, nrb_low=5, exposure='over',
         path='../543.pth', tf="10", dset='sice'):

    test_dataset = build_test_dataset(dset, root_dir, exposure, tf)
    if len(test_dataset) == 0:
        raise RuntimeError(f"No test images found for dset='{dset}' under '{root_dir}'. "
                           "Check that root_dir matches the expected directory layout.")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    lptn_model = LPTNModel(loss_weight, device, lr, gan_type=gan_type,
                           nrb_high=nrb_high, nrb_low=nrb_low, nrb_top=nrb_top,
                           levels=[0, 1, 2], weights=[0.5, 0.3, 0.2])
    lptn_model.load_network(path, device=device)
    lptn_model.net_g.eval()

    psnr_test, ssim_test, lpips_test, mssim_test = 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            lptn_model.feed_data(x, y)
            _, lptn_model.output = lptn_model.net_g(lptn_model.LLI)
            visuals = lptn_model.get_current_visuals()
            psnr_iter, ssim_iter, lpips_iter, mssim_iter = lptn_model.calculate_metrics_test(
                visuals['result'], visuals['High_Limage'])
            psnr_test += psnr_iter
            ssim_test += ssim_iter
            lpips_test += lpips_iter
            mssim_test += mssim_iter
            num_batches += 1

    psnr_test /= num_batches
    ssim_test /= num_batches
    lpips_test /= num_batches
    mssim_test /= num_batches

    print(f'TEST PSNR  ({dset}) {psnr_test}')
    print(f'TEST SSIM  ({dset}) {ssim_test}')
    print(f'TEST LPIPS ({dset}) {lpips_test}')
    print(f'TEST MSSIM ({dset}) {mssim_test}')

    return psnr_test, ssim_test, lpips_test, mssim_test


def eval_model(configs):
    return eval(configs['root_dir'],
                configs['lr'],
                configs['loss_weight'],
                configs['gan_type'],
                configs['device'],
                configs['nrb_top'],
                configs['nrb_high'],
                configs['nrb_low'],
                configs['exposure'],
                configs['model_path'],
                configs['tf'],
                configs['dset'])
