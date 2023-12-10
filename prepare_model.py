import os
import argparse
import torch
import numpy as np
import random
import recovery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare models for exact and approximate unlearning.')
    parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--tr_strat', default='conservative', type=str, help='training strategy')
    parser.add_argument('--epochs', default=None, type=int, help='updated epochs')
    
    parser.add_argument('--seed', default=None, type=int, help='random seed')

    parser.add_argument('--exclude_num', default=0, type=int, help='excluded samples during training')

    parser.add_argument('--save_steps', default=10, type=int, help='steps to save ckpt')

    parser.add_argument('--save_folder', default='results/models', type=str, help='result folder')

    parser.add_argument('--save_name', default='', type=str, help='saving file name')
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
    

    

    save_folder = os.path.join(args.save_folder, f"{args.model.lower()}_{args.dataset.lower()}_ex{args.exclude_num}_{args.save_name}")
    os.makedirs(save_folder, exist_ok=True)
    setup = recovery.utils.system_startup()
    defs = recovery.training_strategy(args.tr_strat)
    defs.validate = args.save_steps

    if args.epochs is not None:
        defs.epochs = args.epochs

    loss_fn, trainloader, validloader, num_classes, excluded_data, data_mean, data_std = recovery.construct_dataloaders(args.dataset, defs, data_path=f'datasets/{args.dataset.lower()}',  shuffle=True, normalize=True, exclude_num=args.exclude_num)


    model, _ = recovery.construct_model(args.model, num_classes=num_classes, seed=args.seed, num_channels=3)
    model.to(**setup)


    stats = recovery.train(model, loss_fn, trainloader, validloader, defs, setup=setup, ckpt_path=save_folder)
    
    resdict = {
        'tr_args': args.__dict__,
        'tr_strat': defs.__dict__,
        'stats': stats,
        'net_sd': model.state_dict(),
        'excluded_data': excluded_data
        }
    torch.save(resdict, os.path.join(save_folder, 'final.pth'))
