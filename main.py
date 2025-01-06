import os
import torch
import wandb
import argparse
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn
from utils import yaml_config_hook, Trainer


def main(args, logger):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.task == '2_cls':
        # binary classification
        wsi_csv = pd.read_csv(args.wsi_csv_path)
        patient_csv = pd.read_csv(args.patient_csv_path)   
        wsi_csv = wsi_csv[wsi_csv['Grade.Revised'].isin(['0', '1'])]
        patient_csv = patient_csv[patient_csv['Grade.Revised'].isin(['0', '1'])]

    elif args.task == '4_cls' or args.task == 'survival':
        # 4 class classification
        wsi_csv = pd.read_csv(args.wsi_csv_path)
        patient_csv = pd.read_csv(args.patient_csv_path)  

    else:
        raise ValueError("task should be one of ['2_cls', '4_cls', 'survival']")

    
    # d_in is depend on the feature extractor
    if args.extractor in ['UNI', 'Kimia', 'Dense121']:
        args.d_in = 1024
    elif args.extractor == 'conch':
        args.d_in = 512
    else:
        raise ValueError("extractor should be one of ['UNI', 'Kimia', 'Dense121', 'conch']")

    patient_list = patient_csv['Case.ID'].values
    patient_label_list = patient_csv['Grade.Revised'].values
    trainer = Trainer(wsi_df=wsi_csv, args=args, wb_logger=logger)
    trainer.kfold_train()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./configs/default.yaml")
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

