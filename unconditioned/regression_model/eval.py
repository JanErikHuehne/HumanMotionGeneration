import os
from os.path import join as pjoin
import numpy as np
import matplotlib.pyplot as plt
from dataloader import create_dataloader
from model import ResNetCNN

import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def generate_sketch(joints, radius=2, name=None):
    kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]

    data = joints * -2.5
    MINS = data.min(axis=0)
    height_offset = MINS[1]
    data[:, 1] -= height_offset
    data[..., 0] -= data[0:1, 0]
    data[..., 2] -= data[0:1, 2]
   

    save_path = "."
    path_root = save_path
    
    save_path = path_root
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([radius/3, 4/3*radius ])
    ax.view_init(elev=160, azim=-90)
    ax.grid(b=False)
    ax.invert_zaxis()
    f = data
    for c in kinematic_tree:
        points = f[c]
        ax.plot3D(points[:, 0], points[:, 2], points[:, 1], linestyle='-', c="black")

        # Show the plot
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.axis('off')
    os.makedirs(pjoin(save_path, "temp"), exist_ok=True)
    if name is None:
        plt.savefig(pjoin(pjoin(save_path, "temp"), f"temp.png"))
    else:
        plt.savefig(pjoin(pjoin(save_path, "temp"), f"{name}.png"))
    plt.clf()
    plt.close()
    return pjoin(save_path, "temp")

def reverse_vector(vec_rep, reference_point=None):
    kinematic_tree = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
    if reference_point is None:
        reference_point = np.zeros((3,))
    rep = rep = {"index": 0, "point" : reference_point}
    points = {0 : reference_point}
    list_vectors = [rep]
    # First we need to extract each vector representation of each respecitive chaib
    # We do this by iterating over the kinematic tree
    # For each chain we extract the vectors and add them to a list
    total_length = 0
    for c in kinematic_tree:
        length = len(c) - 1
        #chain_vectors = np.flip(vec_rep[total_length:total_length+length], axis=0)
        chain_vectors = vec_rep[total_length:total_length+length]
        for i, v in enumerate(chain_vectors):
            start_val = points.get(c[i])
            if start_val is not None:
               
               
                point = start_val + v
                rep = {"index": c[i+1], "point" : point}
                list_vectors.append(rep)

                points[c[i+1]] = point
                #print("Adding new point ", points)
        total_length += length
        #print("--- End of Chain ---")
    point_coord = sorted(list_vectors, key=lambda d: d['index'])
    re_points = np.empty((0,3))
    for p in point_coord:
        re_points = np.append(re_points, p["point"][np.newaxis, ...], axis=0)
    return re_points

if __name__ == "__main__":
    import pickle as pkl
    import pprint
    import tqdm
    pp = pprint.PrettyPrinter(indent=4)
    data = pkl.load(open("/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/humanact12poses.pkl", "rb"))
    data = data["joints3D"]
    kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
    average_joint_length_total = np.zeros(23,)
    for i in tqdm.tqdm(range(len(data))):
        sample = data[i]
        average_joint_length = np.zeros(23,)
        for j in range(len(sample)):
            frame_sample = sample[j]
            transformed_target = np.zeros((23,3))
            i = 0
            for tree in kinematic_tree:
                for j in range(len(tree)-1):
                    transformed_target[i]
                    transformed_target[i] = frame_sample[tree[j+1]] - frame_sample[tree[j]]
                        
                    i += 1
            for i,joint in enumerate(transformed_target):
            
                average_joint_length[i] += np.linalg.norm(joint)

        average_joint_length /= len(sample)
      
        average_joint_length_total += average_joint_length
    average_joint_length_total /= len(data)
    average_joint_length_total = np.round(average_joint_length_total, 2)
    print(average_joint_length_total)
    test_sample = data[1]
    
    test_sample = test_sample[10]

   
    temp_path = generate_sketch(test_sample)

    dataloader = create_dataloader(temp_path, 1, shuffle=False, mode="test")
    model = ResNetCNN().to(device)  # Move model to GPU if available
    model.load_state_dict(torch.load("/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/trained_model.pth", map_location=device))
    for sample in dataloader:
        print("_")
        sample = sample.to(device)
        output = model(sample)
      
        output = output.detach().cpu().numpy()[:]
        output = output.reshape(23, 3)
        joints = reverse_vector(output, reference_point=test_sample[0])
        error = test_sample - joints
        error = np.abs(error).mean()
        print(error)
        generate_sketch(joints, name="test")
