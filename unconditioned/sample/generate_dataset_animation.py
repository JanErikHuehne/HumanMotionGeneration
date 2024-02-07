from data_loaders.humanml.utils.plot_script import plot_3d_motion
import pickle as pkl
import numpy as np
import tqdm
if __name__ == "__main__":
    kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]
    data = pkl.load(open("/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/humanact12poses.pkl", "rb"))
    data = data["joints3D"]
    translation = np.zeros((0,3))
    total_frames = 0
    for d in tqdm.tqdm(data):
        for f in d:
            ft = f[0]
            translation = np.append(translation, ft[np.newaxis,:], axis=0)
    
    print(translation.min(axis=0))
    print(translation.max(axis=0))
    

    #print(translation)
    data = data[26]
    print("REGENERATING MOTION")
    plot_3d_motion(joints=data, kinematic_tree=kinematic_tree,
                    save_path="/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/condtion_/c6/original_motion.mp4",
                    title="Original Motion",
                    dataset="humanact12",
                    fps=20)