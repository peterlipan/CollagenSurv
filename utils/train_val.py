import os
import torch
import warnings
import pandas as pd
from models import CreateModel
from dataset import featureDataset
from .metrics import compute_avg_metrics
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


class MetricLogger:
    def __init__(self, n_folds):
        self.fold = 0
        self.n_folds = n_folds
        self.metrics = ['Accuracy', 'F1', 'AUC', 'AP', 'BAC', 'Sensitivity', 'Specificity', 'Precision', 'MCC', 'Kappa']
        self.fold_metrics = [self._init_dict() for _ in range(n_folds)] # save final metrics for each fold
    
    def _set_fold(self, fold):
        self.fold = fold
    
    def _init_dict(self):
        return {metric: 0.0 for metric in self.metrics}

    def update(self, metric_dict):
        for key in metric_dict:
            self.fold_metrics[self.fold][key] = metric_dict[key]
    
    def _fold_average(self):
        if self.fold < self.n_folds - 1:
            raise Warning("Not all folds have been completed.")
        avg_metrics = self._init_dict()
        for metric in avg_metrics:
            for fold in self.fold_metrics:
                avg_metrics[metric] += fold[metric]
            avg_metrics[metric] /= self.n_folds
        
        return avg_metrics


class Trainer:
    def __init__(self, wsi_df, args, wb_logger=None, val_steps=50, verbose=False):
        self.wsi_df = wsi_df
        self.kfold = args.kfold
        self.args = args
        self.wb_logger = wb_logger
        self.val_steps = val_steps
        self.verbose = verbose
        self.m_logger = MetricLogger(n_folds=self.kfold)
    
    def _dataset_split(self, train_patient_idx, test_patient_idx):
        self.train_dataset = featureDataset(args=self.args, wsi_df=self.wsi_df, patient_idx=train_patient_idx)
        self.test_dataset = featureDataset(args=self.args, wsi_df=self.wsi_df, patient_idx=test_patient_idx)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, 
                                       shuffle=True, num_workers=self.args.workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, 
                                      shuffle=False, num_workers=self.args.workers)
        
        self.args.n_classes = self.train_dataset.num_labels
        self.model = CreateModel(self.args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        
        self.scheduler = None
        if self.args.lr_policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        elif self.args.lr_policy == 'cosine_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        elif self.args.lr_policy == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[18, 19], gamma=0.1)
    

    def kfold_train(self):
        kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.args.seed)
        patient_list = self.wsi_df['Case.ID'].values
        patient_label_list = self.wsi_df['Grade.Revised'].values
        for fold, (train_idx, test_idx) in enumerate(kfold.split(patient_list, patient_label_list)):
            print('-'*20, f'Fold {fold}', '-'*20)
            train_pid = patient_list[train_idx]
            test_pid = patient_list[test_idx]
            self._dataset_split(train_pid, test_pid)

            self.m_logger._set_fold(fold)
            self.fold = fold

            self.train()
            # validate for the fold
            metric_dict = self.validate()
            self.m_logger.update(metric_dict)
            if self.verbose:
                print('-'*20, f'Fold {fold} Metrics', '-'*20)
            print(metric_dict)
        
        avg_metrics = self.m_logger._fold_average()
        print('-'*20, 'Average Metrics', '-'*20)
        print(avg_metrics)
        self._save_fold_avg_results(avg_metrics)


    def train(self):
        self.model.train()
        cur_iters = 0
        for i, (features, labels) in enumerate(self.train_loader):
            features, labels = features.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    
            outputs = self.model(features)
            loss = self.criterion(outputs.logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            cur_iters += 1
            if cur_iters % self.val_steps == 0:
                cur_lr = self.optimizer.param_groups[0]['lr']
                metric_dict = self.validate()
                if self.verbose:
                    print(f"Fold {self.fold} | Epoch {i} | Loss: {loss.item()} | Acc: {metric_dict['Accuracy']} | LR: {cur_lr}")
                if self.wb_logger is not None:
                    self.wb_logger.log({f"Fold_{self.fold}": {
                        'Train': {'loss': loss.item(), 'lr': cur_lr},
                        'Test': metric_dict
                    }})

    def validate(self):
        training = self.model.training
        self.model.eval()
        ground_truth = torch.Tensor().cuda()
        probabilities = torch.Tensor().cuda()
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                outputs = self.model(features)
                prob = outputs.y_prob

                ground_truth = torch.cat((ground_truth, labels), dim=0)
                probabilities = torch.cat((probabilities, prob), dim=0)
            
            metric_dict = compute_avg_metrics(ground_truth, probabilities, avg=self.args.metric_avg)
        
        self.model.train(training)

        return metric_dict

    def _save_fold_avg_results(self, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold
        task2name = {'2_cls': 'Binary', '4_cls': '4Class', 'survival': 'Survival'}
        taskname = task2name[self.args.task]

        df_name = f"{self.args.kfold}Fold_{taskname}.xlsx"
        res_path = self.args.results
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        settings = ['Model', 'KFold', 'Feature Extractor', 'Magnification', 'Patch Size', 
                    'Patch Overlap', 'New Annotation', 'Stain Normalization', 'Augmentation', 'Metric Average Method']
        kwargs = ['backbone', 'kfold', 'extractor', 'magnification', 'patch_size', 'patch_overlap', 'calibrate', 'stain_norm', 'augmentation', 'metric_avg']
        set2kwargs = {k: v for k, v in zip(settings, kwargs)}

        metric_names = self.m_logger.metrics
        df_columns = settings + metric_names
        
        df_path = os.path.join(res_path, df_name)
        if not os.path.exists(df_path):
            df = pd.DataFrame(columns=df_columns)
        else:
            df = pd.read_excel(df_path)
            if df_columns != df.columns.tolist():
                warnings.warn("Columns in the existing excel file do not match the current settings.")
                df = pd.DataFrame(columns=df_columns)
        
        new_row = {k: self.args.__dict__[v] for k, v in set2kwargs.items()}
        # fine-grained modification for better presentation
        new_row['Feature Extractor'] = new_row['Feature Extractor'].upper()
        new_row['Magnification'] = f"{new_row['Magnification']}\u00D7"
        new_row['New Annotation'] = 'Yes' if new_row['New Annotation'] else 'No'
        new_row['Stain Normalization'] = 'Yes' if new_row['Stain Normalization'] else 'No'
        new_row['Augmentation'] = 'Yes' if new_row['Augmentation'] else 'No'

        if keep_best: # keep the rows with the best mcc for each fold
            exsiting_rows = df[(df[settings] == pd.Series(new_row)).all(axis=1)]
            if not exsiting_rows.empty:
                exsiting_mcc = exsiting_rows['MCC'].values
                if metric_dict['MCC'] > exsiting_mcc:
                    df = df.drop(exsiting_rows.index)
                else:
                    return

        new_row.update(metric_dict)
        df = df._append(new_row, ignore_index=True)
        df.to_excel(df_path, index=False)
        
