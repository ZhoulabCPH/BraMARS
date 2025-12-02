import torch
import sys
import numpy as np
import os
import pandas as pd
import warnings
import random
from torch.utils.data import DataLoader
from torch import nn
import torch.backends.cudnn as cudnn
from S4MIL import S4Model
from patch import cut_patch
from extractor import get_features_CTransPath

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


def calculate_metrics(model, data_loader, thresholds):
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        data = pd.DataFrame()
        model.eval()
        for step, slide in enumerate(data_loader):
            slide = slide.cuda()
            pred_ = model(slide)
            score = sigmoid(pred_).detach().cpu().numpy()
        if score > thresholds:
            preds = 1
        else:
            preds = 0
        data['SCORES'] = score
        data['PREDS'] = preds

    return data


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


def eval_(root, seeds):
    slide_root = root
    slide_name = root.split('\\')[-1].split('.')[0]
    seed = seeds
    setup_seed(seed)
    print('WSI to patch start!')
    cut_patch.tiling_WSI(slide_root)
    get_features_CTransPath.get_feature(slide_root)
    features_root = os.path.join(slide_root.split('.')[0], 'features')
    features_paths = os.path.join(features_root, slide_name) + '.pkl'
    features = torch.load(features_paths)['features']
    data_loader = DataLoader(features, batch_size=1, shuffle=True, collate_fn=collatte_fn)
    setup_seed(seed)
    path = "model_wight_root"  # model_wight_root
    model = S4Model(in_dim=768, n_classes=1, act='gelu', dropout=False).cuda()
    ckpt = torch.load(path, map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    wsi_data = calculate_metrics(model, data_loader, ckpt['threshold'])
    if wsi_data['PREDS'] == 1:
        print('BraMARS recommendations: High risk')
    else:
        print('BraMARS recommendations: Low risk')


if __name__ == '__main__':
    seeds = 9197
    root = 'WSI_root'  # WSI storage address
    sys.stdout = open('log_root', 'w')  # Log storage address
    print('BraMARS start!')
    eval_(root, seeds)
    sys.stdout.close()
