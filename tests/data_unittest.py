import unittest
import json 
import torch
import os
from data_loaders.utils.opt import get_opt
from data_loaders import dataset


class TestDataloader(unittest.TestCase):    
    def test_opt(self):
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        opt = get_opt(opt_path =os.path.abspath("./tests/test_data/opt_test.txt"), device=device)
        assert opt.data_root == "dataset/HumanML3D"
        assert opt.motion_dir == "dataset/HumanML3D/new_joint_vecs"
        
    def test_dataset(self):
        splits = ["train", "test", "val"]
        expected_lengths = [336726, 63084, 21166]
        for split, el in zip(splits, expected_lengths): 
            dt = dataset.HumanML3D(datapath="tests/test_data/opt_test.txt", split=split)
            assert len(dt) == el, "Expected length {} of {} dataset, found ".format(el,split) + str(len(dt))

if __name__ == "__main__":
    unittest.main()



