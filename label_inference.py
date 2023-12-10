import os
import argparse
import torch
import copy
from torchvision import transforms
import numpy as np
import pickle
import recovery
import pandas as pd


def make_dict(tmp_dict, output, labels, name):
    if tmp_dict is None:
        tmp_dict = {'Confidence':[], 'Type':[], 'Class':[]}

    for c in range(output.shape[1]):
        idx_per_c = torch.where(labels == c)[0][:10]
        conf_list = output[idx_per_c, c].tolist()

        tmp_dict['Confidence'].extend(conf_list)
        tmp_dict['Type'].extend([name] * len(conf_list))
        tmp_dict['Class'].extend([c] * len(conf_list))
    return tmp_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--load_folder_name', type=str, default='resnet18_stl10_ex1000_s0', help='folder of pretrained model')
    parser.add_argument('--model_save_folder', type=str, default='results/models', help='folder of pretrained model')
    args = parser.parse_args()

    model = 'ResNet18'
    dataset = 'stl10'
    num_classes = 10
    seed = 0 
    ft_samples = 512
    img_size = 32 if dataset == 'cifar10' else 96
    excluded_num = 10000 if dataset == 'cifar10' else 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    final_dict = torch.load(os.path.join(args.model_save_folder, args.load_folder_name, 'final.pth'))
    setup = recovery.utils.system_startup()
    defs = recovery.training_strategy('conservative')
    defs.lr = 1e-3
    defs.epochs = 1
    defs.batch_size = 128
    defs.optimizer = 'SGD'
    defs.scheduler = 'linear'
    defs.warmup = False
    defs.weight_decay  = 0.0
    defs.dropout = 0.0
    defs.augmentations = False
    defs.dryrun = False


    loss_fn, _org_trainloader, validloader, num_classes, _exd, dmlist, dslist =  recovery.construct_dataloaders(dataset.lower(), defs, data_path=f'datasets/{dataset.lower()}', normalize=False, exclude_num=excluded_num)
    dm = torch.as_tensor(dmlist, **setup)[:, None, None]
    ds = torch.as_tensor(dslist, **setup)[:, None, None]
    normalizer = transforms.Normalize(dmlist, dslist)


    # *** used for batch case ***
    excluded_data = final_dict['excluded_data']
    index = torch.tensor(np.random.choice(len(excluded_data[0]), ft_samples, replace=False))
    print("Batch index", index.tolist())
    X_all, y_all = excluded_data[0][index], excluded_data[1][index]
    print("FT data size", X_all.shape, y_all.shape)
    trainset_all = recovery.data_processing.SubTrainDataset(X_all, y_all, transform=transforms.Normalize(dmlist, dslist))
    trainloader_all = torch.utils.data.DataLoader(trainset_all, batch_size=min(defs.batch_size, len(trainset_all)), shuffle=True,  num_workers=8, pin_memory=True)



    model_pretrain, _ = recovery.construct_model(model, num_classes=num_classes, num_channels=3)
    model_pretrain.load_state_dict(final_dict['net_sd'])
    model_pretrain.eval()

    ft_folder = os.path.join(args.model_save_folder, args.load_folder_name, 'probing_samples')
    ft_path = os.path.join(ft_folder, f'finetune_{defs.epochs}ep.pt')

    model_ft, _ = recovery.construct_model(model, num_classes=num_classes, num_channels=3)
    model_ft.load_state_dict(torch.load(ft_path))
    model_ft.eval()


    class_pred_dict = {}
    for class_id in range(num_classes):
        print(class_id)
        # ## Unlearn one


        model_ft = model_ft.cpu()
        unlearn_ids = torch.where(y_all == class_id)[0]
        unlearn_ids = unlearn_ids[:len(unlearn_ids) // 10]
        print(f"Unlearn {unlearn_ids}")

        X = torch.stack([xt for i, xt in enumerate(X_all) if i not in unlearn_ids])
        y = torch.tensor([yt for i, yt in enumerate(y_all) if i not in unlearn_ids])
        print("Exact unlearn data size", X.shape, y.shape)
        trainset_unlearn = recovery.data_processing.SubTrainDataset(X, y, transform=transforms.Normalize(dmlist, dslist))
        trainloader_unlearn = torch.utils.data.DataLoader(trainset_unlearn, batch_size=min(defs.batch_size, len(trainset_unlearn)), shuffle=True, num_workers=8, pin_memory=True)

        X_unlearn = torch.stack([xt for i, xt in enumerate(X_all) if i in unlearn_ids])
        y_unlearn = torch.tensor([yt for i, yt in enumerate(y_all) if i in unlearn_ids])

        print(f"***** Train unlearned model (withouth {unlearn_ids}) *****")
        model_unlearn_one, _ = recovery.construct_model(model, num_classes=num_classes, num_channels=3)
        model_unlearn_one.load_state_dict(final_dict['net_sd'])
        model_unlearn_one.eval()
        model_unlearn_one.to(**setup)
        unlearn_stats = recovery.train(model_unlearn_one, loss_fn, trainloader_unlearn, validloader, defs, setup=setup, ckpt_path=None, finetune=True)
        model_unlearn_one.cpu()

        # approx
        batch_size = min(defs.batch_size, len(X_unlearn))
        approx_diff = [p.detach() for p in recovery.recovery_algo.loss_steps(model_ft, normalizer(X_unlearn), y_unlearn, lr=defs.lr, local_steps=defs.epochs * len(X_unlearn) // batch_size, batch_size=batch_size)] 
        model_app_unlearn_one = copy.deepcopy(model_ft)

        old_params = {}
        for i, (name, params) in enumerate(model_app_unlearn_one.named_parameters()):
            old_params[name] = params.clone()
            old_params[name] += approx_diff[i]
        for name, params in model_app_unlearn_one.named_parameters():
            params.data.copy_(old_params[name])


        # ## unlearn half


        unlearn_ids = torch.where(y_all == class_id)[0]
        unlearn_ids = unlearn_ids[:len(unlearn_ids) // 2]
        print(f"Unlearn {unlearn_ids}")

        X = torch.stack([xt for i, xt in enumerate(X_all) if i not in unlearn_ids])
        y = torch.tensor([yt for i, yt in enumerate(y_all) if i not in unlearn_ids])
        print("Exact unlearn data size", X.shape, y.shape)
        trainset_unlearn = recovery.data_processing.SubTrainDataset(X, y, transform=transforms.Normalize(dmlist, dslist))
        trainloader_unlearn = torch.utils.data.DataLoader(trainset_unlearn, batch_size=min(defs.batch_size, len(trainset_unlearn)), shuffle=True, num_workers=8, pin_memory=True)

        X_unlearn = torch.stack([xt for i, xt in enumerate(X_all) if i in unlearn_ids])
        y_unlearn = torch.tensor([yt for i, yt in enumerate(y_all) if i in unlearn_ids])

        print(f"***** Train unlearned model (withouth {unlearn_ids}) *****")
        model_unlearn_half, _ = recovery.construct_model(model, num_classes=num_classes, num_channels=3)
        model_unlearn_half.load_state_dict(final_dict['net_sd'])
        model_unlearn_half.eval()
        model_unlearn_half.to(**setup)
        unlearn_stats = recovery.train(model_unlearn_half, loss_fn, trainloader_unlearn, validloader, defs, setup=setup, ckpt_path=None, finetune=True)
        model_unlearn_half.cpu()

        # approx
        batch_size = min(defs.batch_size, len(X_unlearn))
        approx_diff = [p.detach() for p in recovery.recovery_algo.loss_steps(model_ft, normalizer(X_unlearn), y_unlearn, lr=defs.lr, local_steps=defs.epochs * len(X_unlearn) // batch_size, batch_size=batch_size)] 
        model_app_unlearn_half = copy.deepcopy(model_ft)

        old_params = {}
        for i, (name, params) in enumerate(model_app_unlearn_half.named_parameters()):
            old_params[name] = params.clone()
            old_params[name] += approx_diff[i]
        for name, params in model_app_unlearn_half.named_parameters():
            params.data.copy_(old_params[name])


        # ## unlearn all


        unlearn_ids = torch.where(y_all == class_id)[0]
        print(f"Unlearn {unlearn_ids}")

        X = torch.stack([xt for i, xt in enumerate(X_all) if i not in unlearn_ids])
        y = torch.tensor([yt for i, yt in enumerate(y_all) if i not in unlearn_ids])
        print("Exact unlearn data size", X.shape, y.shape)
        trainset_unlearn = recovery.data_processing.SubTrainDataset(X, y, transform=transforms.Normalize(dmlist, dslist))
        trainloader_unlearn = torch.utils.data.DataLoader(trainset_unlearn, batch_size=min(defs.batch_size, len(trainset_unlearn)), shuffle=True, num_workers=8, pin_memory=True)

        X_unlearn = torch.stack([xt for i, xt in enumerate(X_all) if i in unlearn_ids])
        y_unlearn = torch.tensor([yt for i, yt in enumerate(y_all) if i in unlearn_ids])

        print(f"***** Train unlearned model (withouth {unlearn_ids}) *****")
        model_unlearn, _ = recovery.construct_model(model, num_classes=num_classes, num_channels=3)
        model_unlearn.load_state_dict(final_dict['net_sd'])
        model_unlearn.eval()
        model_unlearn.to(**setup)
        unlearn_stats = recovery.train(model_unlearn, loss_fn, trainloader_unlearn, validloader, defs, setup=setup, ckpt_path=None, finetune=True)
        model_unlearn.cpu()

        # approx
        batch_size = min(defs.batch_size, len(X_unlearn))
        approx_diff = [p.detach() for p in recovery.recovery_algo.loss_steps(model_ft, normalizer(X_unlearn), y_unlearn, lr=defs.lr, local_steps=defs.epochs * len(X_unlearn) // batch_size, batch_size=batch_size)] # lr is not important in cosine 
        model_app_unlearn = copy.deepcopy(model_ft)
        old_params = {}
        for i, (name, params) in enumerate(model_app_unlearn.named_parameters()):
            old_params[name] = params.clone()
            old_params[name] += approx_diff[i]
        for name, params in model_app_unlearn.named_parameters():
            params.data.copy_(old_params[name])


        # # Plot


        sample_dict = pickle.load(open(os.path.join(args.model_save_folder, args.load_folder_name, 'probing_samples','query_sample_dict.pkl'), 'rb'))
        sample_datats = torch.cat([x[:10] for x in sample_dict.values()])


        with torch.no_grad():
            model_ft.cuda()
            test_output = model_ft(normalizer(sample_datats.cuda())).softmax(dim=1).detach().cpu() 
            model_ft.cpu()
            sample_labelts = test_output.argmax(dim=1)
            
            model_unlearn_one.cuda()
            exact_unlearn_class1_one =model_unlearn_one(normalizer(sample_datats.cuda())).detach().softmax(dim=1).cpu() 
            model_unlearn_one.cpu()
            
            model_app_unlearn_one.cuda()
            approx_unlearn_class1_one =model_app_unlearn_one(normalizer(sample_datats.cuda())).softmax(dim=1).cpu() 
            model_app_unlearn_one.cpu()
            
            model_unlearn_half.cuda()
            exact_unlearn_class1_half =model_unlearn_half(normalizer(sample_datats.cuda())).softmax(dim=1).cpu() 
            model_unlearn_half.cpu()
            
            model_app_unlearn_half.cuda()
            approx_unlearn_class1_half =model_app_unlearn_half(normalizer(sample_datats.cuda())).softmax(dim=1).cpu() 
            model_app_unlearn_half.cpu()
            
            model_unlearn.cuda()
            exact_unlearn_class1 =model_unlearn(normalizer(sample_datats.cuda())).softmax(dim=1).cpu()
            model_unlearn.cpu()
            
            model_app_unlearn.cuda()
            approx_unlearn_class1 =model_app_unlearn(normalizer(sample_datats.cuda())).softmax(dim=1).cpu()
            model_app_unlearn.cpu()
            

        v, i = test_output.max(dim=1)
        print(v, i, np.unique(i, return_counts=True))


        all_dict = None
        all_dict = make_dict(all_dict, exact_unlearn_class1_one - test_output, sample_labelts, 'Exact $(p_u=0.1)$')
        all_dict = make_dict(all_dict, approx_unlearn_class1_one - test_output, sample_labelts, 'Approx. $(p_u=0.1)$')
        all_dict = make_dict(all_dict, exact_unlearn_class1_half - test_output, sample_labelts, 'Exact $(p_u=0.5)$')
        all_dict = make_dict(all_dict, approx_unlearn_class1_half - test_output, sample_labelts, 'Approx. $(p_u=0.5)$')
        all_dict = make_dict(all_dict, exact_unlearn_class1 - test_output, sample_labelts, 'Exact $(p_u=1.0)$')
        all_dict = make_dict(all_dict, approx_unlearn_class1 - test_output, sample_labelts, 'Approx. $(p_u=1.0)$')
        plot_df = pd.DataFrame.from_dict(all_dict)



        tmp = plot_df.groupby(['Type', 'Class']).agg({'Confidence': 'mean'}).reset_index()

        tmp['CC'] = tmp[['Class', 'Confidence']].apply(tuple, axis=1)

        class_pred_dict[class_id] = tmp.groupby('Type')['CC'].agg(lambda x: list(x)[np.array([x0[1] for x0 in list(x)]).argmin()][0])
    
    print(class_pred_dict)
    pickle.dump(class_pred_dict, open(os.path.join(args.model_save_folder, f'{dataset}_unlearnlabel_{defs.epochs}.pkl'), 'wb'))

