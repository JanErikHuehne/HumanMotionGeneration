import torch
from torch.utils import data
import numpy as np
import os 
from os.path import join as pjoin
import random
from data_loaders.utils.opt import get_opt
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
        self.motion_length = 41
        self.data_dict = {}
        id_list = []
        new_name_list = []
        with open(data_file, "r") as f:
            if top_10:
                for line in f.readlines():
                    if len(id_list) < 10:
                        id_list.append(line.strip())
                    else:
                        break
            else: 
                for line in f.readlines():
                    id_list.append(line.strip())

        self.data_dict2 = []
        self.data_dict3 = []
        for name in tqdm(id_list):
            motion = pjoin(opt.motion_dir, name + '.npy')
            motion_length = np.load(motion).shape[0]
            if motion_length < self.motion_length:   # why is there a condition? why we need to select motion whoes length is bigger than self.motion_length
                continue
            sketch_dir = pjoin(opt.condition_root, name)
            sketch_data_paths = os.listdir(sketch_dir)
            sketches = []
            for sketch in sketch_data_paths:
                if str(sketch).split("_")[-1].split(".")[-1] == 'npy':
                    i = 0
                    sketch_frame = str(sketch).split("_")[-1].split(".")[0]

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
                    sketches.append((pjoin(sketch_dir, sketch), int(sketch_frame)))  # JC change
                    new_name_list.append(new_name)
            sketches.sort(key=lambda x: x[1])
            # for i in range(motion_length // 10 - 4):

            i = 0
            while i * 10 + 40 < motion_length:
            # for i in range(3)[-1:]:
                sketch_crop = [sketches[i], sketches[i+1], sketches[i+2], sketches[i+3], sketches[i+4]]
                motion_crop = (i*10, i*10+40)
                self.data_dict3.append({
                    'motion': motion,
                    'sketches': sketch_crop,
                    'motion_crop': motion_crop,
                })
                i += 4


            """
            # idx = np.round(np.linspace(0, len(sketches) - 1, 5)).astype(int)
            # selected = []

            # Define the range as a list from which to pick numbers
            # For example, to pick from numbers 1 through 10
            range_of_numbers = list(range(1, len(sketches)-1))  # This creates a list from 1 to len(sketches)-1

            # Use random.sample to pick 3 unique values
            random.seed(5)
            idx = random.sample(range_of_numbers, 3)

            # idx = np.round(np.linspace(0, len(sketches) - 1, 5)).astype(int)
            idx.append(0)
            idx.append(len(sketches) - 1)
            selected = []

            for id in idx:
                selected.append(sketches[id])

            
            sketches = selected

            self.data_dict3.append({
                    'motion': motion,
                    'sketches': sketches,
                    #'motion_crop': motion_crop,
                })
            """
            self.data_dict2.append({'motion': motion,
                                    'sketches': sketches
            })
        name_list = sorted(new_name_list)
        self.name_list = name_list

    def inverse_norm(self, data):
        return data * self.std + self.mean

    # def __len__(self):
    #     return len(self.data_dict)

    def __len__(self):
        return len(self.data_dict3)
    
    def transform_img(self, img):
        """
        Data Augmentation Transform 
        img: PIL Image object
        """
        return self.processor(images=img, return_tensors="pt", padding=True)
      
       
    def __getitem__(self, index):
        """
        motion with fixed length of 40 frames
        3 sketches for each motion
        interval of sketches is 20 frames
        """
        data = self.data_dict3[index]
        motion = np.load(data["motion"])
        motion = (motion - self.mean) / self.std
        sketches = []
        key_frames = []
        sketches_tuple = data["sketches"]
        random.shuffle(sketches_tuple)
        for sketch_name, key_frame in sketches_tuple:
            # sketch = Image.open(sketch_name)
            sketch = np.load(sketch_name.split(".")[0] + '.npy')
            sketches.append(sketch)
            key_frames.append(key_frame)
        key_frames = torch.tensor(key_frames)
        # sketches = self.transform_img(sketches)
        sketches = torch.tensor(np.array(sketches))
        sketches = torch.squeeze(sketches)
        #motion = motion[self.data_dict3[index]['motion_crop'][0]:self.data_dict3[index]['motion_crop'][1]+1]
        # sketches = torch.stack(sketches, dim=1)
        # sketches = torch.squeeze(sketches)


        return motion, sketches, key_frames



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
        opt.motion_dir = pjoin(opt.data_root, "new_joints_vec")
        # opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.save_root = pjoin(abs_base_path, 'save')
        self.opt = opt

        print('Loading dataset %s ....' % opt.dataset_name)
        print(f'dataset_root: {opt.condition_root}, {opt.motion_dir}')
        self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        self.t2m_dataset = Sketch2MotionDataset(self.mean, self.std, self.opt, self.split_file)


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)
        
    def __len__(self):
        return self.t2m_dataset.__len__()
        
