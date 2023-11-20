import torch
from torch.utils import data
import numpy as np
import os 
from os.path import join as pjoin
import random
from .utils.opt import get_opt 
import logging
from tqdm import tqdm
from PIL import Image
from PIL.ImageOps import grayscale
from torchvision.transforms import GaussianBlur, RandomRotation, ToTensor, Compose

class Sketch2MotionDataset(data.Dataset):

    def __init__(self, mean, std, opt, data_file, transform=True):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.transform = transform
        self.data_dict = {}
        id_list = []
        new_name_list = []
        with open(data_file, "r") as f:
            for line in f.readlines():

                id_list.append(line.strip())


        for name in tqdm(id_list):
            motion = pjoin(opt.motion_dir, name + '.npy')
            sketch_dir = pjoin(opt.condition_root, name)
            sketch_data_paths = os.listdir(sketch_dir)
            for sketch in sketch_data_paths:
                i = 0
                new_name = str(i) + "_" + name
                while new_name in self.data_dict:
                    i += 1
                    try:
                        new_name = str(i) + '_' + name
                    except Exception:
                        print(len(sketch_data_paths))
                        print("Too many files!")
                self.data_dict[new_name] = {'motion': motion,
                                       'sketch': sketch}
                new_name_list.append(new_name)
        name_list = sorted(new_name_list)
        self.name_list = name_list

    def inverse_norm(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)
    
    def transform(self, img):
        """
        Data Augmentation Transform 
        img: PIL Image object
        """
        img = img.crop((200, 200, 800, 800))
        img = img.resize((200,200))
        img = grayscale(img)

        transforms = Compose([
            RandomRotation([-20, 20], fill=255),
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
            ToTensor()
            ]
            )
        return transforms(img)


    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, sketch = np.load(data["motion"]), Image.open(data["sketch"])
        m_length = motion.shape[0]
        m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length

        idx = random.randint(0, len(motion)- m_length)
        motion = motion[idx:idx+m_length]

        # Z Norm 
        motion = (motion -self.mean) / self.std
        sketch = self.transform(sketch)
        


class HumanML3D(data.Dataset):
    def __init__(self, split="train", datapath="", **kwargs):
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = f"."

        dataset_opt_path = pjoin(abs_base_path, datapath)
        print(dataset_opt_path)
        opt = get_opt(dataset_opt_path, device=None)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.condition_root = pjoin(opt.data_root, "sketches")
        opt.motion_dir = pjoin(opt.data_root,"new_joint_vecs")
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt

        print('Loading dataset %s ....' % opt.dataset_name)
        self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        self.t2m_dataset = Sketch2MotionDataset(self.mean, self.std, self.opt, self.split_file)


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)
        
    def __len__(self):
        return self.t2m_dataset.__len__()
        