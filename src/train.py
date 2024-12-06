import argparse
from utils.trainer import train_model
import wandb


def main(args):
    config = {
        'rgb_dir': args.rgb_dir,
        'th_low_res_dir': args.th_low_res_dir,
        'tar_dir': args.tar_dir,
        'rgb_val_dir': args.rgb_val_dir,
        'th_low_res_val_dir': args.th_low_res_val_dir,
        'tar_val_dir': args.tar_val_dir,
        'rgb_test_dir': args.rgb_test_dir,
        'th_low_res_test_dir': args.th_low_res_test_dir,
        'tar_test_dir':args.tar_test_dir,
        'rgb_dir1':args.rgb_dir1,
        'tar_dir1':args.tar_dir1,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        # 'seed': args.seed,
        'device':args.device,
        'lr': args.lr,
        'loss_weight':args.loss_weight,
        'dataset':args.dataset,
        'gan_type':args.gan_type,
        'nrb_low':args.nrb_low,
        'nrb_high':args.nrb_high,
        'cats_directory': args.cats_directory,
        'rgb_uav_dir':args.rgb_uav_dir,
        'thermal_low_res_uav_dir': args.thermal_low_res_uav_dir,
        'thermal_high_res_uav_dir': args.thermal_high_res_uav_dir,
        'nrb_top': args.nrb_top
    }

    wandb.init(project="Lamar-new", entity="kasliwal17",
               config={'lr':args.lr, 'max_ssim':0, 'max_psnr':0,'test_psnr':0,'test_ssim': 0, 'best_epoch':0, 'nrb_low':args.nrb_low, 'nrb_high':args.nrb_high, 'loss_weight':args.loss_weight, 'gan_type': args.gan_type, 'nrb_top': args.nrb_top, 'dataset': args.dataset}, allow_val_change=True)
    train_model(config)

if __name__ == '__main__':

    ##Detailed description of the arguments can be found in the README.md file
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, required=False, default='.Dataset/ULB/Train/RGB_Train')
    parser.add_argument('--th_low_res_dir', type=str, required=False, default='.Dataset/ULB/Train/Thermal_Low_Res_Train')
    parser.add_argument('--tar_dir', type=str, required=False, default='.Dataset/ULB/Train/Thermal_High_Res_Train')
    parser.add_argument('--rgb_val_dir', type=str, required=False, default='.Dataset/ULB/Validation/RGB_Validation')
    parser.add_argument('--th_low_res_val_dir', type=str, required=False, default='.Dataset/ULB/Validation/Thermal_Low_Res_Validation')
    parser.add_argument('--tar_val_dir', type=str, required=False, default='.Dataset/ULB/Validation/Thermal_High_Res_Validation')
    parser.add_argument('--rgb_test_dir', type=str, required=False, default='.Dataset/ULB/Test/RGB_Test')
    parser.add_argument('--th_low_res_test_dir', type=str, required=False, default='.Dataset/ULB/Test/Thermal_Low_Res_Test')
    parser.add_argument('--tar_test_dir', type=str, required=False, default='.Dataset/ULB/Test/Thermal_High_Res_Test')
    parser.add_argument('--rgb_dir1', type=str, required=False, default='./FLIR_ADAS_v2/video_rgb_test/data')
    parser.add_argument('--tar_dir1', type=str, required=False, default='./FLIR_ADAS_v2/video_thermal_test/data')
    parser.add_argument('--batch_size', type=int, required=False, default=12)
    parser.add_argument('--epochs', type=int, required=False, default=300)
    
    # parser.add_argument('--seed', type=int, required=False, default=3)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--loss_weight', type=float, required=False, default=2000)
    parser.add_argument('--dataset', type=str, required=True, default='UAV')
    parser.add_argument('--gan_type', type=str, required=False, default='lsgan')
    parser.add_argument('--nrb_low', type=int, required=False, default=3)
    parser.add_argument('--nrb_high', type=int, required=False, default=5)
    parser.add_argument('--cats_directory', type=str, required=False, default='/kaggle/input/cats-dataset')
    parser.add_argument('--rgb_uav_dir', type=str, required=False, default= '/kaggle/input/vgtsr-uav/VGTSR/HR RGB')
    parser.add_argument('--thermal_low_res_uav_dir', type=str, required=False, default='/kaggle/input/vgtsr-uav/VGTSR/LR thermal/4/BI')
    parser.add_argument('--thermal_high_res_uav_dir', type=str, required=False, default='/kaggle/input/vgtsr-uav/VGTSR/GT thermal')
    parser.add_argument('--nrb_top', type=int, required=False, default=4)
    arguments = parser.parse_args()
    main(arguments)
