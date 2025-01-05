import os
import torch
import wandb
import argparse
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn
from models import CreateModel
from utils import yaml_config_hook
from qmh.dataset.dataset import featureDataset
from torch.utils.data import DataLoader
from train_val import train, fold_validate
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def main(args, logger):


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.task == '2_cls':
        # binary classification
        wsi_csv = pd.read_csv(args.wsi_csv_path)
        patient_csv = pd.read_csv(args.patient_csv_path)   
        wsi_csv = wsi_csv[wsi_csv['Grade.Revised'].isin(['0', '1'])]
        patient_csv = patient_csv[patient_csv['Grade.Revised'].isin(['0', '1'])]

    elif args.task == '4_cls':
        # 4 class classification
        wsi_csv = pd.read_csv(args.wsi_csv_path)
        patient_csv = pd.read_csv(args.patient_csv_path)  

    elif args.task == 'survival':
        #TODO: read data for survival task
        pass

    else:
        raise ValueError("task should be one of ['2_cls', '4_cls', 'survival']")


    patient_list = patient_csv['Case.ID'].values
    patient_label_list = patient_csv['Grade.Revised'].values


    fold_num = 0
    all_fold_auc = []
    all_fold_auc_macro = []
    all_fold_acc = []
    all_fold_gt_list = []
    all_fold_pred_list = []
    kfold = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for train_idx, test_idx in kfold.split(patient_list, patient_label_list):
        # for train 
        train_patient_list = patient_list[train_idx]
        train_patient_label = patient_label_list[train_idx]
        # for test
        test_patient_list = patient_list[test_idx]
        test_patient_label = patient_label_list[test_idx]

        fold_num += 1
        args.fold_num = fold_num
        print(f"\nfold: {fold_num}")

        best_auc_val_fold = 0.0
        best_acc_val_fold = 0.0
        all_loss_train= []

        print("preparing datasets and dataloaders......")
        train_dataset = featureDataset(args=args, all_wsi_info=wsi_csv, patient_idx=train_patient_list)
        test_dataset = featureDataset(args=args, wsi_df=wsi_csv, patient_idx=test_patient_list)
        # num of labels
        num_labels = train_dataset.num_labels

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True)
        loaders = (train_loader, test_loader)

        print(f"Train Case num: {len(train_idx)}, Test Case num: {len(test_idx)}")
        print(f"Train WSI num: {len(train_loader)}, Test WSI num: {len(test_loader)}")

        model = CreateModel(args).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.lr_policy == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        elif args.lr_policy == 'cosine_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5)
        elif args.lr_policy == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[18,19], gamma=0.1)
        else:
            scheduler = 'None'

        # train and val
        train(model, loaders, criterion, opt, scheduler, logger, args)
        # fold validation
        fold_auc, fold_auc_macro, fold_acc, fold_gt_list, fold_pred_list = fold_validate(model, test_loader, logger, args)
        all_fold_auc.append(fold_auc)
        all_fold_auc_macro.append(fold_auc_macro)
        all_fold_acc.append(fold_acc)

        all_fold_gt_list.extend(fold_gt_list)
        all_fold_pred_list.extend(fold_pred_list)

    # Calculate mean
    mean_fold_auc = np.mean(all_fold_auc)
    mean_fold_auc_macro = np.mean(all_fold_auc_macro)
    mean_fold_acc = np.mean(all_fold_acc)
    # Calculate variance
    # variance = np.var(accuracies)
    var_fold_auc = np.var(all_fold_auc)
    var_fold_auc_macro = np.var(all_fold_auc_macro)
    var_fold_acc = np.var(all_fold_acc)
    print(f"\nmean fold auc: {mean_fold_auc}, mean fold auc_macro: {mean_fold_auc_macro}, mean fold acc: {mean_fold_acc}")
    print(f"var fold auc: {var_fold_auc}, var fold auc_macro: {var_fold_auc_macro}, var fold acc: {var_fold_acc}")

    if args.save4CM == 'all_fold':
        # print('\n2 list len:', len(gt_list)) #17 for all test set in one fold
        confusion_for_val = confusion_matrix(all_fold_gt_list, all_fold_pred_list)
        print(f'All folds_confusion_matrix_val:{confusion_for_val}')
        ConfusionMatrixDisplay(confusion_for_val).plot()
        if not os.path.exists('./outputs/Confusion_Matrix'):
            os.makedirs('./outputs/Confusion_Matrix')
            print("Confusion_Matrix Folder created successfully.")
        plt.savefig('./outputs/Confusion_Matrix/all_fold.jpg')
        # plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    if not args.debug:
        wandb.login(key="5a217af8c48db3869e827b99d99fdf6a6330f04e")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="QMH",
            config=config
        )
    else:
        wandb_logger = None


    main(args, wandb_logger)

