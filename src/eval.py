import argparse
from utils.evaluater import eval_model
import wandb


def main(args):
    config = {
        'batch_size': args.batch_size,
        'root_dir' : args.root_dir,
        'epochs': args.epochs,
        'device':args.device,
        'kernel_loss_weight':args.kernel_loss_weight,
        'lr': args.lr,
        'dset': args.dset,
        'loss_weight':args.loss_weight,
        'gan_type':args.gan_type,
        'use_hypernet': args.use_hypernet,
        'sf_path' : args.sf_path,
        'nrb_top' :args.nrb_top,
        'nrb_high' : args.nrb_high,
        'nrb_low' : args.nrb_low
    }

    eval_model(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--root_dir', type=str, required=False)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--dset', type=str, required=False, default='grad')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--loss_weight', type=float, required=False, default=2000)
    parser.add_argument('--kernel_loss_weight', type=float, required=False, default=100)
    parser.add_argument('--gan_type', type=str, required=False)
    parser.add_argument('--use_hypernet', type=bool, required=False, default=True)
    parser.add_argument('--key', type=str, required=False, default='3adf824888485fb1de047a4e9bab54143ddf0cd9')
    parser.add_argument('--sf_path', type=str, required=False, default='./best_model_g.pth')
    parser.add_argument('--nrb_low', type=int, required=False)
    parser.add_argument('--nrb_high', type=int, required=False)
    parser.add_argument('--nrb_top', type=int, required=False)
    arguments = parser.parse_args()
    main(arguments)
