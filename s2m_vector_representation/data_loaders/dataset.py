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

    def __init__(self, mean, std, opt, data_file, transform=True, top_10 = True):
        self.mean = mean
        self.std = std
        self.opt = opt
        model_name = "openai/clip-vit-base-patch16"
        # self.processor = CLIPProcessor.from_pretrained(model_name)
        self.transform = transform
        self.motion_length = 40
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
            vector_dir = pjoin(opt.condition_root, name)
            vector_data_paths = os.listdir(vector_dir)
            vectors = []
            for vector in vector_data_paths:
                i = 0
                vector_frame = str(vector).split("_")[1]
                
                new_name = str(i) + "_" + name
                while new_name in self.data_dict:
                    i += 1
                    try:
                        new_name = str(i) + '_' + name
                    except Exception:
                        print(len(vector_data_paths))
                        print("Too many files!")
                self.data_dict[new_name] = {'motion': motion,
                                       'vector': pjoin(vector_dir, vector),
                                       'vector_frame': int(vector_frame)}
                vectors.append((pjoin(vector_dir, vector), int(vector_frame)))
                new_name_list.append(new_name)
            vectors.sort(key=lambda x: x[1])
            for i in range(motion_length // 10 - 4):
                vector_crop = [vectors[i], vectors[i+1], vectors[i+2], vectors[i+3], vectors[i+4]]
                motion_crop = (i*10, i*10+40)
                self.data_dict3.append({
                    'motion': motion,
                    'vectors': vector_crop,
                    'motion_crop' : motion_crop,
                })
            self.data_dict2.append( {'motion': motion,
                                       'vectors': vectors
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


        # data = self.data_dict[self.name_list[item]]
        # motion, sketch = np.load(data["motion"]), Image.open(data["sketch"])
        # max_length = motion.shape[0]
        # # Check if half of the motion size fits after the sketch
        # half_length = int((self.motion_length/2))
        # fits_after = (max_length-data["sketch_frame"]-1) -half_length
        # fits_before = data["sketch_frame"] - half_length
        # # Create a window around the frame
        # if fits_after > 0 and fits_before > 0:
        #     motion = motion[data["sketch_frame"]-half_length:data["sketch_frame"]+half_length]
        # # If window does not fit in front, we talk available frames in front and will the
        # # overlay with frames from the behind the sketch
        # elif fits_after > 0 and fits_before < 0:
        #     motion = motion[:self.motion_length]
        # # If the window does not fit behind, we take available frames from in front of the frame
        # else:
        #     motion = motion[-self.motion_length:]
        #
        #
        # m_length = motion.shape[0]
        # m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        #
        # idx = random.randint(0, len(motion)- m_length)
        # motion = motion[idx:idx+m_length]
    def __getitem__(self, index):
        """
        motion with fixed length of 40 frames
        3 sketches for each motion
        interval of sketches is 20 frames
        """
        data = self.data_dict3[index]
        motion = np.load(data["motion"])
        motion = (motion - self.mean) / self.std
        # vectors = []
        # key_frames = []
        # for vector_name, key_frame in data["vectors"]:
        #     vector = np.load(vector_name)
        #     vectors.append(vector)
        #     key_frames.append(key_frame)
        # key_frames, vectors = torch.tensor(key_frames), torch.tensor(np.array(vectors))
        key_frames, vectors = torch.tensor([0, 10, 20, 30, 40]), torch.tensor(np.array([np.zeros([21, 2]), np.zeros([21, 2]), np.zeros([21, 2]), np.zeros([21, 2]), np.zeros([21, 2])]))

        # sketches = self.transform_img(sketches)
        motion = motion[self.data_dict3[index]['motion_crop'][0]:(self.data_dict3[index]['motion_crop'][1] + 1)]
        # sketches = torch.stack(sketches, dim=1)
        # sketches = torch.squeeze(sketches)


        return motion, vectors, key_frames



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
        opt.condition_root = pjoin(opt.data_root, "Vectors")
        opt.motion_dir = pjoin(opt.data_root,"new_joints_vec")
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
        
