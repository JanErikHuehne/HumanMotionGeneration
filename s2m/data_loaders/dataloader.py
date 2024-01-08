
from data_loaders.dataset import HumanML3D
from torch.utils.data import DataLoader
import torch as th
# import torch
from transformers import CLIPModel
def dev():
    """
    Get the device to use.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:0")
    return th.device("cpu")
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
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    for i, (motion, vectors, key_frames) in enumerate(loader):
        # sketch.to(dev())
        vectors = vectors.to(device)
        # sketches['pixel_values'] = sketches['pixel_values'].reshape(-1, 3, 224, 224)
        # outputs = model.get_image_features(**sketches)
        # print(outputs.shape)
        # print(outputs)
        # outputs = outputs.reshape(-1, 5, 512)
        # zero_indices = (vectors != 1).nonzero(as_tuple=True)
        # print(zero_indices)
        # print(vectors.shape)

if __name__ == "__main__":
    main()