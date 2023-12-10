
from data_loaders.dataset import HumanML3D
from torch.utils.data import DataLoader

def get_dataset_loader(datapath, batch_size, split='train'):
    if split == "train":
        shuffle = True
    else: 
        shuffle = False
    dataset = HumanML3D(split=split, datapath=datapath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader

def main():

    loader = get_dataset_loader(datapath='test_data/humanml_opt.txt', batch_size=1, split="train")
    for i, (motion, sketch) in enumerate(loader):
        print(motion.shape)
        zero_indices = (sketch != 1).nonzero(as_tuple=True)
        print(zero_indices)
        print(sketch.shape)

if __name__ == "__main__":
    main()