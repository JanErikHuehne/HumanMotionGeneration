import numpy as np
import pickle as pkl 
from eval import generate_sketch, reverse_vector
data = pkl.load(open("/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/humanact12poses.pkl", "rb"))
data = data["joints3D"]
kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]

data = data[1]
data = data[0]

data = data - data[0,:]
print(data)
JOINT_LENGTHS = np.array([0.11,
                 0.36,
                 0.38,
                 0.14,
                 0.11,
                 0.36,
                 0.38,
                 0.13,
                 0.12,
                 0.13,
                 0.05,
                 0.22,
                 0.1,
                 0.14,
                 0.12,
                 0.25,
                 0.25,
                 0.09,
                 0.14,
                 0.12,
                 0.25,
                 0.25,
                 0.09], dtype=float)
JOINT_LENGTHS = JOINT_LENGTHS[..., np.newaxis]



generate_sketch(data, name="from_data")


 
kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
transformed_target = np.zeros((23,3))
i = 0
target = data

for tree in kinematic_tree:
    for j in range(len(tree)-1):
        transformed_target[i]
        transformed_target[i] = target[tree[j+1]] - target[tree[j]]
                
        #transformed_target[i] = transformed_target[i] / np.linalg.norm(transformed_target[i])
     
        i += 1

target = transformed_target
#target = target * JOINT_LENGTHS

target = reverse_vector(target)
generate_sketch(target, name="after")