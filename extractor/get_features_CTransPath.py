import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath
import os

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)


class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        path_num = os.path.basename(path)

        return image, path_num


def get_feature(root):
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load('ctranspath.pth')  # 官方ctranspath.pth文件地址
    model.load_state_dict(td['model'], strict=True)
    roots = root.split('.')[0]
    slide_name = root.split('\\')[-1].split('.')[0]
    patch_root = os.path.join(roots, 'patch')
    file_path = patch_root  # 储存patch的文件夹的路径
    save_path = os.path.join(roots, 'features')  # 特征存放的文件夹的路径
    img_path = os.listdir(file_path)
    for idx in range(len(img_path)):
        img_path[idx] = os.path.join(file_path, img_path[idx])
    test_datat = roi_dataset(img_path)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        i = 0
        a = []
        b = []
        for batch, path_num in database_loader:
            features = model(batch)
            features = features.cpu().numpy()
            features = np.squeeze(features)
            a.append(features)
            b.append(path_num)
            i += 1
            # print(i)
        a = torch.Tensor(a)
        data = {
            'features': a,
            'path_name': b
        }
        paths = os.path.join(save_path, slide_name) + '.pkl'
        torch.save(data, paths)

    print("finish")

