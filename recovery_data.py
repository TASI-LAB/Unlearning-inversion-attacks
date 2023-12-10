import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import recovery as rs
import argparse


recons_config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.04, 
              optim='adamw',
              restarts=5,
              max_iterations=10000,
              total_variation=1e-2,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

def new_plot(tensor, title="", path=None):
    if tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(2 * tensor.shape[0], 3))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.savefig(path)

def process_recons_results(result, ground_truth, figpath, recons_path, filename):
    output_list, stats, history_list, x_optimal = result
    x_optimal = x_optimal.detach().cpu()
    test_mse = (x_optimal - ground_truth.cpu()).pow(2).mean()
    test_psnr = rs.metrics.psnr(x_optimal, ground_truth, factor=1/ds)
    title = f"MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | "
    new_plot(torch.cat([ground_truth, x_optimal]), title, path=os.path.join(figpath, f'{filename}.png'))
    torch.save({'output_list': output_list.cpu(), 'stats': stats, 'history_list': history_list, 'x_optimal': x_optimal}, open(os.path.join(recons_path, f'{filename}.pth'), 'wb'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple argparse.')
    parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--ft_samples', default=32, type=int)
    parser.add_argument('--unlearn_samples', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int, help='updated epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model_save_folder', default='results/models', type=str, help='folder of pretrained models')

    args = parser.parse_args()

    print(args.__dict__)

    img_size = 32 if 'cifar' in args.dataset else 96
    excluded_num = 10000 if 'cifar' in args.dataset else 1000
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    load_folder_name = f'{args.model.lower()}_{args.dataset.lower()}_ex{excluded_num}_s0'
    save_folder_name = f'ex{args.ft_samples}_un{args.unlearn_samples}_ep{args.epochs}_seed{args.seed}'
    save_folder = os.path.join(args.model_save_folder, load_folder_name, save_folder_name)
    os.makedirs(save_folder, exist_ok=True)
    
    final_dict = torch.load(os.path.join(args.model_save_folder, load_folder_name, 'final.pth'))
    setup = rs.utils.system_startup()
    defs = rs.training_strategy('conservative')
    defs.lr = args.lr
    defs.epochs = args.epochs
    defs.batch_size = 128
    defs.optimizer = 'SGD'
    defs.scheduler = 'linear'
    defs.warmup = False
    defs.weight_decay  = 0.0
    defs.dropout = 0.0
    defs.augmentations = False
    defs.dryrun = False

    
    loss_fn, _tl, validloader, num_classes, _exd, dmlist, dslist =  rs.construct_dataloaders(args.dataset.lower(), defs, data_path=f'datasets/{args.dataset.lower()}', normalize=False, exclude_num=excluded_num)
    dm = torch.as_tensor(dmlist, **setup)[:, None, None]
    ds = torch.as_tensor(dslist, **setup)[:, None, None]
    normalizer = transforms.Normalize(dmlist, dslist)


    # *** used for batch case ***
    excluded_data = final_dict['excluded_data']
    index = torch.tensor(np.random.choice(len(excluded_data[0]), args.ft_samples, replace=False))
    print("Batch index", index.tolist())
    X_all, y_all = excluded_data[0][index], excluded_data[1][index]
    print("FT data size", X_all.shape, y_all.shape)
    trainset_all = rs.data_processing.SubTrainDataset(X_all, y_all, transform=transforms.Normalize(dmlist, dslist))
    trainloader_all = torch.utils.data.DataLoader(trainset_all, batch_size=min(defs.batch_size, len(trainset_all)), shuffle=True,  num_workers=8, pin_memory=True)
    

    ## load state dict
    state_dict =  final_dict['net_sd']


    model_pretrain, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
    model_pretrain.load_state_dict(state_dict)
    model_pretrain.eval()
    
    
    model_ft, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
    model_ft.load_state_dict(state_dict)
    model_ft.eval()


    print("Train full model.")
    ft_folder = os.path.join(save_folder, 'full_ft')
    os.makedirs(ft_folder, exist_ok=True)
    model_ft.to(**setup)
    ft_stats = rs.train(model_ft, loss_fn, trainloader_all, validloader, defs, setup=setup, ckpt_path=ft_folder, finetune=True)
    model_ft.cpu()
    resdict = {'tr_args': args.__dict__,
        'tr_strat': defs.__dict__,
        'stats': ft_stats,
        'batch_index': index,
        'train_data': (X_all, y_all)}
    torch.save(resdict, os.path.join(ft_folder, 'finetune_params.pth'))
    ft_diffs = [(ft_param.detach().cpu() - org_param.detach().cpu()).detach() for (ft_param, org_param) in zip(model_ft.parameters(), model_pretrain.parameters())]








    print("Exact unlearn each sample and test the exact and approximate unlearn")
    
    model_ft.zero_grad()
    model_ft.to(**setup)
    rec_machine_ft = rs.GradientReconstructor(model_ft, (dm, ds), recons_config, num_images=args.unlearn_samples)
    
    model_pretrain.zero_grad()
    model_pretrain.to(**setup)
    rec_machine_pretrain = rs.GradientReconstructor(model_pretrain, (dm, ds), recons_config, num_images=args.unlearn_samples)
    

    for test_id in range(args.ft_samples // args.unlearn_samples):
        unlearn_ids = list(range(test_id * args.unlearn_samples, (test_id + 1) * args.unlearn_samples))
        print(f"Unlearn {unlearn_ids}")
        unlearn_folder = os.path.join(save_folder, f'unlearn_ft_batch{test_id}')
        os.makedirs(unlearn_folder, exist_ok=True)
        X_list = [xt for i, xt in enumerate(X_all) if i not in unlearn_ids]
        if len(X_list) > 0:
            X = torch.stack([xt for i, xt in enumerate(X_all) if i not in unlearn_ids])
            y = torch.tensor([yt for i, yt in enumerate(y_all) if i not in unlearn_ids])
            print("Exact unlearn data size", X.shape, y.shape)
            trainset_unlearn = rs.data_processing.SubTrainDataset(X, y, transform=transforms.Normalize(dmlist, dslist))
            trainloader_unlearn = torch.utils.data.DataLoader(trainset_unlearn, batch_size=min(defs.batch_size, len(trainset_unlearn)), shuffle=True, num_workers=8, pin_memory=True)
        
        X_unlearn = torch.stack([xt for i, xt in enumerate(X_all) if i in unlearn_ids])
        y_unlearn = torch.tensor([yt for i, yt in enumerate(y_all) if i in unlearn_ids])

        print(f"***** Train unlearned model (withouth {unlearn_ids}) *****")
        model_unlearn, _ = rs.construct_model(args.model, num_classes=num_classes, num_channels=3)
        model_unlearn.load_state_dict(state_dict)
        model_unlearn.eval()
        model_unlearn.to(**setup)
        if len(X_list) > 0:
            unlearn_stats = rs.train(model_unlearn, loss_fn, trainloader_unlearn, validloader, defs, setup=setup, ckpt_path=unlearn_folder, finetune=True)
        else:
            unlearn_stats = None
        model_unlearn.cpu()
        resdict = {'tr_args': args.__dict__,
            'tr_strat': defs.__dict__,
            'stats': unlearn_stats,
            'unlearn_batch_id': test_id}
        torch.save(resdict, os.path.join(unlearn_folder, 'finetune_params.pth'))
        # unlearn_params =  [param.detach() for param in model_unlearn.parameters()]
        un_diffs = [(un_param.detach().cpu() - org_param.detach().cpu()).detach() for (un_param, org_param) in zip(model_unlearn.parameters(), model_pretrain.parameters())]

        print("Start reconstruction.")
        


        recons_folder = os.path.join(save_folder, 'recons')
        figure_folder = os.path.join(save_folder, 'figures')
        os.makedirs(recons_folder , exist_ok=True)
        os.makedirs(figure_folder, exist_ok=True)
        # reconstruction
        
        
        exact_diff = [-(ft_diff * args.ft_samples - un_diff * len(X_list)).detach().to(**setup) for (ft_diff, un_diff) in zip(ft_diffs, un_diffs)]
        rec_machine_pretrain.model.eval()
        result_exact = rec_machine_pretrain.reconstruct(exact_diff, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), img_shape=(3, img_size, img_size))
        process_recons_results(result_exact, X_unlearn, figpath=figure_folder, recons_path=recons_folder, filename=f'exact{test_id}_{index[test_id].item()}')

        approx_diff = [p.detach().to(**setup) for p in rs.recovery_algo.loss_steps(model_ft, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), lr=1, local_steps=1)] # lr is not important in cosine 
        rec_machine_ft.model.eval()
        result_approx = rec_machine_ft.reconstruct(approx_diff, normalizer(X_unlearn.to(**setup)), y_unlearn.to(setup['device']), img_shape=(3, img_size, img_size))
        process_recons_results(result_approx, X_unlearn, figpath=figure_folder, recons_path=recons_folder, filename=f'approx{test_id}_{index[test_id].item()}')

        




        