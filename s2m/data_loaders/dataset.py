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
from transformers import CLIPProcessor
from torchvision.transforms import GaussianBlur, RandomRotation, ToTensor, Compose

class Sketch2MotionDataset(data.Dataset):

    def __init__(self, mean, std, opt, data_file, transform=True, top_10 = False):
        self.mean = mean
        self.std = std
        self.opt = opt
        model_name = "openai/clip-vit-base-patch16"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.transform = transform
        self.motion_length=60
        self.data_dict = {}
        id_list = []
        new_name_list = []
        with open(data_file, "r") as f:
            if top_10:
                for line in f.readlines() :
                    if len(id_list) < 10:
                        id_list.append(line.strip())
            else: 
                for line in f.readlines() :
                    id_list.append(line.strip())

        self.data_dict2 = []
        for name in tqdm(id_list):
            motion = pjoin(opt.motion_dir, name + '.npy')
            motion_length = np.load(motion).shape[0]
            if motion_length < self.motion_length:   # why is there a condition? why we need to select motion whoes length is bigger than self.motion_length
                continue
            sketch_dir = pjoin(opt.condition_root, name)
            sketch_data_paths = os.listdir(sketch_dir)
            sketches = []
            for sketch in sketch_data_paths:
                i = 0
                sketch_frame = str(sketch).split("_")[-1].split(".png")[0]
                
                new_name = str(i) + "_" + name
                while new_name in self.data_dict:
                    i += 1
                    try:
                        new_name = str(i) + '_' + name
                    except Exception:
                        print(len(sketch_data_paths))
                        print("Too many files!")
                self.data_dict[new_name] = {'motion': motion,
                                       'sketch': pjoin(sketch_dir, sketch),
                                       'sketch_frame': int(sketch_frame)}
                sketches.append((pjoin(sketch_dir, sketch), int(sketch_frame))) # JC change
                new_name_list.append(new_name)
            sketches.sort(key=lambda x: x[1])
            self.data_dict2.append( {'motion': motion,
                                       'sketches': sketches
            })
        name_list = sorted(new_name_list)
        self.name_list = name_list

    def inverse_norm(self, data):
        return data * self.std + self.mean

    # def __len__(self):
    #     return len(self.data_dict)

    def __len__(self):
        return len(self.data_dict2)
    
    def transform_img(self, img):
        """
        Data Augmentation Transform 
        img: PIL Image object
        """
        return self.processor(images=image, return_tensors="pt", padding=True)
      
       


    # def __getitem__(self, item, **kwargs):
    #     data = self.data_dict[self.name_list[item]]
    #     # data0 = self.data_dict2[str(item)]
    #     motion, sketch = np.load(data["motion"]), Image.open(data["sketch"])
    #     # for sketch0 in sketch:
    #     max_length = motion.shape[0]
    #     # Check if half of the motion size fits after the sketch
    #     half_length = int((self.motion_length/2))
    #     fits_after = (max_length-data["sketch_frame"]-1) -half_length
    #     fits_before = data["sketch_frame"] - half_length
    #     # Create a window around the frame
    #     if fits_after > 0 and fits_before > 0:
    #         motion = motion[data["sketch_frame"]-half_length:data["sketch_frame"]+half_length]
    #     # If window does not fit in front, we talk available frames in front and will the
    #     # overlay with frames from the behind the sketch
    #     elif fits_after > 0 and fits_before < 0:
    #         motion = motion[:self.motion_length]
    #     # If the window does not fit behind, we take available frames from in front of the frame
    #     else:
    #         motion = motion[-self.motion_length:]
    #
    #
    #     m_length = motion.shape[0]
    #     m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
    #
    #     idx = random.randint(0, len(motion)- m_length)
    #     motion = motion[idx:idx+m_length]
    #
    #     # Z Norm
    #     motion = (motion -self.mean) / self.std
    #     sketch = self.transform_img(sketch)
    #     return motion, sketch

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, sketch = np.load(data["motion"]), Image.open(data["sketch"])
        max_length = motion.shape[0]
        # Check if half of the motion size fits after the sketch
        half_length = int((self.motion_length/2))
        fits_after = (max_length-data["sketch_frame"]-1) -half_length 
        fits_before = data["sketch_frame"] - half_length
        # Create a window around the frame
        if fits_after > 0 and fits_before > 0: 
            motion = motion[data["sketch_frame"]-half_length:data["sketch_frame"]+half_length]
        # If window does not fit in front, we talk available frames in front and will the 
        # overlay with frames from the behind the sketch
        elif fits_after > 0 and fits_before < 0:
            motion = motion[:self.motion_length]
        # If the window does not fit behind, we take available frames from in front of the frame
        else: 
            motion = motion[-self.motion_length:]
    
            
        m_length = motion.shape[0]
        m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length

        idx = random.randint(0, len(motion)- m_length)
        motion = motion[idx:idx+m_length]

        # Z Norm 
        motion = (motion -self.mean) / self.std
        sketch = self.transform_img(sketch)
        return motion, sketch



class HumanML3D(data.Dataset):
    def __init__(self, split="train", datapath="", **kwargs):
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = f"."

        dataset_opt_path = pjoin(abs_base_path, datapath)
        print(dataset_opt_path)
        opt = get_opt(dataset_opt_path, device=None)
        # opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.data_root = os.path.abspath(pjoin(abs_base_path, opt.data_root))
        opt.condition_root = pjoin(opt.data_root, "sketches")
        opt.motion_dir = pjoin(opt.data_root,"new_joint_vecs")
        # opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.save_root = pjoin(abs_base_path, 'save')
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
        
