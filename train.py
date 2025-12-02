import torch
import sys
import numpy as np
import os
import pandas as pd
from datasets import FeatureDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import random
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
from sklearn.model_selection import StratifiedKFold
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter
import lifelines
from S4MIL import S4Model

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def calculate_acc(data_loader):
    pred_list = data_loader['PREDS'].tolist()
    label_list = data_loader['LABELS'].tolist()
    acc = 0
    for idx_ in range(len(pred_list)):
        if pred_list[idx_] == label_list[idx_]:
            acc += 1
    acc = acc / len(pred_list)
    return acc


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False


def get_args(lr, wd):
    parser = argparse.ArgumentParser(description='BM invasion prediction')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--checkpoint-dir', default='', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=lr)
    parser.add_argument('--wd', default=wd)
    parser.add_argument('--T_max', default=10)
    args = parser.parse_args()
    return args


def calculate_metrics(model, data_loader, thresholds):
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        data = pd.DataFrame()
        model.eval()
        score = np.array([])
        label = []
        slides = []
        for step, (slide, slide_property) in enumerate(data_loader):
            slide = slide.cuda()
            slides = slides + slide_property['slide_name']
            pred_ = model(slide)
            score = np.append(score, (sigmoid(pred_).detach().cpu().numpy()))
            label = label + list(slide_property['label'].detach().cpu().numpy())

        auc = roc_auc_score(label, score)
        dfs_ = []
        os_ = []
        dfs_state = []
        os_tate = []
        for ik in slides:
            dfs_.append(dfs_os[ik]['DFS'])
            os_.append(dfs_os[ik]['OS'])
            dfs_state.append(dfs_os[ik]['DFS State'])
            os_tate.append(dfs_os[ik]['OS State'])

        preds = (score > thresholds).astype(float)
        Surival_Data = pd.DataFrame()
        Surival_Data['DFS'] = dfs_
        Surival_Data['Label'] = preds
        Surival_Data['DFS State'] = dfs_state
        Surival_Data['OS'] = os_
        Surival_Data['OS State'] = os_tate

        ##
        result0 = multivariate_logrank_test(Surival_Data['DFS'], Surival_Data['Label'], Surival_Data['DFS State'])
        logrank_p0_2ways = result0.p_value
        Surival_Data_High_Lows = Surival_Data.dropna(axis=0)
        try:
            cph = CoxPHFitter()
            cph.fit(Surival_Data_High_Lows[['DFS', 'DFS State', 'Label']], 'DFS', event_col='DFS State')
        except lifelines.exceptions.ConvergenceError:
            hr_values = 100

        data['SLIDES'] = list(slides)
        data['LABELS'] = list(label)
        data['SCORES'] = list(score)
        data['PREDS'] = list(preds)

    return auc, data, logrank_p0_2ways


def collatte_fn(batch):
    max_len = max(len(features) for features, slide in batch)
    batch_features = []
    batch_labels = []
    batch_name = []

    for features, slide in batch:
        padded_features = torch.zeros((max_len, 768), dtype=torch.float32)
        padded_features[:len(features), :] = features
        name = slide['slide_name']
        label = slide['label']

        batch_features.append(padded_features)
        batch_labels.append(label)
        batch_name.append(name)

    batch_features = torch.stack(batch_features)
    slide_property = {}
    slide_property['slide_name'] = batch_name
    slide_property['label'] = torch.tensor(batch_labels)

    return batch_features, slide_property


def train(lr, wd, warmup, seed):
    args = get_args(lr, wd)
    seed = seed
    setup_seed(seed)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    '''读取数据集路径和构建dataloader'''
    # External validation set
    External_val_root = ''  # External_val特征存放地址
    External_val_csv = pd.read_csv('')  # 需要用到的External_val患者数据
    External_val_path = []
    External_val_label_path = []
    for i in range(len(External_val_csv)):
        External_val_path.append(External_val_root + str(External_val_csv['Name'][i]))
        External_val_label_path.append(External_val_csv['Label'][i])
    data_External_val = FeatureDataset(External_val_path, External_val_label_path, mode='test')
    data_External_val_loader = DataLoader(data_External_val, batch_size=1, shuffle=True, collate_fn=collatte_fn)
    # Train set and Internal validation set
    label_csv = pd.read_csv('')  # 需要用到的Train_set患者数据
    root = ''  # Train_set特征存放地址
    list_train_path = []
    label_train = []
    for i in range(len(label_csv)):
        list_train_path.append(root + str(label_csv['Name'][i]))
        label_train.append(label_csv['Label'][i])

    setup_seed(seed)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    a, b, c = enumerate(skf.split(list_train_path, label_train))
    train_idx, val_idx = b[1][0], b[1][1]
    setup_seed(seed)
    fold_train, fold_val = [list_train_path[i] for i in train_idx], [list_train_path[i] for i in val_idx]
    fold_train_label, fold_val_label = [label_train[i] for i in train_idx], [label_train[i] for i in val_idx]

    model = S4Model(in_dim=768, n_classes=1, act='gelu', dropout=0).cuda()

    l_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).cuda())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup, after_scheduler=scheduler)
    data_train = FeatureDataset(fold_train, fold_train_label, mode='train')
    data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, collate_fn=collatte_fn)
    data_validation = FeatureDataset(fold_val, fold_val_label, mode='val')
    data_validation_loader = torch.utils.data.DataLoader(data_validation, batch_size=1, shuffle=False,
                                                         collate_fn=collatte_fn)

    early_stop = 0
    auc_validation = 0
    epoch_validation = 0
    m = t = 0
    for epoch in range(args.epochs):
        early_stop = early_stop + 1
        if early_stop > 20:
            print('Early stop!')
            break
        LOSS = []
        score_ = np.array([])
        LABEL_ = []
        model.train()
        progress_bar = tqdm(total=len(data_train_loader), desc=" BM_invasion training")
        for step, (slide, slide_property) in enumerate(data_train_loader, start=epoch * len(data_train_loader)):
            optimizer.zero_grad()
            slide = slide.cuda()
            label = slide_property['label'].cuda().to(torch.float64)
            pred = model(slide)
            pred = pred.reshape(-1)
            sigmoid = nn.Sigmoid()
            score_ = np.append(score_, (sigmoid(pred).detach().cpu().numpy()))
            LABEL_ = LABEL_ + list(slide_property['label'].detach().cpu().numpy())
            loss1 = l_bce(pred, label)
            loss = loss1
            loss.backward()
            LOSS.append(loss.item())
            optimizer.step()
            progress_bar.update()

        progress_bar.close()
        train_auc = roc_auc_score(LABEL_, score_)
        fpr, tpr, thresholds = roc_curve(LABEL_, score_)

        for idx in range(len(thresholds)):
            if tpr[idx] - fpr[idx] > m:
                m = abs(-fpr[idx] + tpr[idx])
                t = thresholds[idx]

        print(f'thresholds:{t}')

        train_acc_list = (score_ > t).astype(float)
        acc_sum = 0
        for q in range(len(train_acc_list)):
            if train_acc_list[q] == LABEL_[q]:
                acc_sum += 1
        train_acc = acc_sum / len(train_acc_list)
        auc_validation_, data_validation, p_value = calculate_metrics(model, data_validation_loader, t)

        if auc_validation_ > auc_validation:
            auc_validation = auc_validation_
            epoch_validation = epoch
            early_stop = 0
            state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(), threshold=t)
            torch.save(state, args.checkpoint_dir / 'checkpoint_max_validation_auc.pth')
            data_validation.to_csv(args.checkpoint_dir / 'temp_report_validation.csv')
        print('Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(LOSS)))
        print(f'Epoch: {epoch} train AUC: {train_auc}    train ACC: {train_acc}')
        print(f'Epoch: {epoch} validation AUC: {auc_validation_}  validation P: {p_value}')
        print(f'Current best validation AUC: {auc_validation} at epoch {epoch_validation}')
        auc_External_val, data_External_val, p_External_val = calculate_metrics(model, data_External_val_loader, t)
        print(f'Epoch: {epoch},  External_val AUC: {auc_External_val}, External_val P: {p_External_val}')
        print('-------------------------------------------------------------------------------------------')
        scheduler.step()

    model = S4Model(in_dim=768, n_classes=1, act='gelu', dropout=0).cuda()

    ckpt = torch.load(args.checkpoint_dir / f'checkpoint_max_validation_auc.pth', map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    auc_validation, data_validation, p_validation = calculate_metrics(model, data_validation_loader, ckpt['threshold'])
    auc_External_val, data_External_val, p_External_val = calculate_metrics(model, data_External_val_loader,
                                                                            ckpt['threshold'])
    th = ckpt['threshold']

    print(f'Epoch: {epoch} Validation AUC: {auc_validation}, External_val AUC: {auc_External_val}, threshold: {th}')
    data_validation.to_csv(args.checkpoint_dir / f'seed={seed}/temp_report_validation.csv')
    data_External_val.to_csv(args.checkpoint_dir / f'seed={seed}/temp_report_External_val.csv')
    print('Finish!')


if __name__ == '__main__':
    survival = pd.read_csv('')  # 患者临床信息
    dfs_os = {}
    for i in range(survival.shape[0]):
        row_data = survival.loc[i]
        dfs_os[str(row_data['Name'])] = row_data

    lrs = 1e-5
    wds = 1e-4
    warmups = 40
    seeds = 9197
    sys.stdout = open('log_root', 'w')  # 日志存放地址
    print(f'lr {lrs}, wd {wds} , warmup {warmups} , seed {seeds} begin')
    train(lrs, wds, warmups, seeds)
    sys.stdout.close()
