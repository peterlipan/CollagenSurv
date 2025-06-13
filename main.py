import os
import random
import torch
import wandb
import argparse
import warnings
import pandas as pd
import numpy as np
from utils import yaml_config_hook, Trainer


def all_paths_exist(row, root):
    # Path with original extension for path1
    path1 = os.path.join(root, row['Folder'], row['Filename'])
    # Paths with .png extension for path2 and path3
    filename_png = os.path.splitext(row['Filename'])[0] + '.png'
    path2 = os.path.join(root, f"{row['Folder']}_HDM", filename_png)
    path3 = os.path.join(root, f"{row['Folder']}_Masks", filename_png)
    return os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3)

def main(args, logger):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_df = pd.read_excel(args.image_df_path)

    # clean up the DataFrame if the image cannot be found
    image_df = image_df[image_df.apply(lambda row: all_paths_exist(row, args.image_root), axis=1)]

    trainer = Trainer(image_df=image_df, args=args, wb_logger=logger)
    trainer.kfold_train(args)


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
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="ColleganSurv",
            config=config
        )
    else:
        wandb_logger = None

    main(args, wandb_logger)

