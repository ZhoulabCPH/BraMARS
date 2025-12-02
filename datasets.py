from torch.utils.data import Dataset
import torch

class FeatureDataset(Dataset):
    def __init__(self, slides_path, label, mode):
        self.slide = slides_path
        self.label = label
        self.mode = mode

    def __len__(self):
        return len(self.slide)

    def __getitem__(self, index):
        label = self.label[index]
        label = int(label)
        slide = self.slide[index]
        slide_name = slide.split('/')[-1]
        features = torch.load(self.slide[index] + '.pkl')['features']
        slide_property = {}
        slide_property['slide_name'] = slide_name
        slide_property['label'] = torch.tensor(label)
        return features, slide_property

