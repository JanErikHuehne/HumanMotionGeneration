import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle as pkl
import numpy as np 


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = np.random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle)


def transform_img(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),  # Add random rotation transform
        transforms.ToTensor()
    ])
    return transform(image)


class ImageDataset(Dataset):
    def __init__(self, root_dir, ttype="poses", mode="train"):
        self.ttype = ttype
        self.root_dir = root_dir
        self.sketch_dir = os.path.join(root_dir, "sketches")
        self.act12_data = os.path.join(root_dir, "humanact12poses.pkl")
        with open(self.act12_data, "rb") as f:
            self.act12_data = pkl.load(f)
        if ttype == "joints":
            self.joints = self.act12_data["joints3D"]
        elif ttype == "poses":
            self.joints = self.act12_data["poses"]
            self.mean = np.load(os.path.join(root_dir, "mean.npy"))[np.newaxis, ...]
            self.std = np.load(os.path.join(root_dir, "std.npy"))[np.newaxis, ...]
            #for i in range(len(self.joints)):
            #self.joints[i] = (self.joints[i] - self.mean) / self.std
        if mode == "train":
            train_file = os.path.join(root_dir, "train.txt")
            with open(train_file, "r") as f:
                self.image_files = f.readlines()
                self.image_files = [i.strip() for i in self.image_files]
                for i in self.image_files:
                    if "\n" in i:
                        print(i)
        elif mode =="test":
            test_file = os.path.join(root_dir, "test.txt")

            with open(test_file, "r") as f:
                self.image_files = f.readlines()
            self.image_files = [i.strip() for i in self.image_files]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),  # Add random rotation transform
            transforms.ToTensor()
        ])
            

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = os.path.join(self.sketch_dir, self.image_files[idx])
        file_id = self.image_files[idx].split("_")[0]
        frame_id = self.image_files[idx].split("_")[1].split(".")[0]
        image = Image.open(image_path).convert("L")
     
        image = self.transform(image)
        if True:
            target = self.joints[int(file_id)][int(frame_id)]
            if self.ttype == "joints":
                kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
                transformed_target = np.zeros((23,3))
                i = 0
                for tree in kinematic_tree:
                    for j in range(len(tree)-1):
                        transformed_target[i]
                        transformed_target[i] = target[tree[j+1]] - target[tree[j]]
                    
                        i += 1
        
                target = transformed_target / np.linalg.norm(transformed_target, axis=1)[:, np.newaxis]
                    
                target = torch.Tensor(target)
            else:
                target = torch.Tensor(target)
            
      
        return image, target, [file_id, frame_id]



def create_dataloader(dataset_path, batch_size, shuffle=True, ttype="poses", mode="train"):


    dataset = ImageDataset(dataset_path, mode=mode, ttype=ttype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
   
    return dataloader


if __name__ == "__main__":
    sketch_dir = "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/sketches"
    image_files = os.listdir(sketch_dir)
    train, test = torch.utils.data.random_split(image_files, [0.9, 0.1])

    train = list(train)
    print(len(test))
    print(len(train))
    path = "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/train.txt"
    with open(path, "w") as f:
        for t in train:
            f.write(t + "\n")
    
    path = "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/test.txt"
    with open(path, "w") as f:
        for t in test:
            f.write(t + "\n")