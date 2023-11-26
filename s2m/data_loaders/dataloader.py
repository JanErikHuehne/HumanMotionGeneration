
from .dataset import HumanML3D
from torch.utils.data import DataLoader

def get_dataset_loader(datapath, batch_size, split='train'):
    if split == "train":
        shuffle = True
    else: 
        shuffle = False
    dataset = HumanML3D(split=split, datapath=datapath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader