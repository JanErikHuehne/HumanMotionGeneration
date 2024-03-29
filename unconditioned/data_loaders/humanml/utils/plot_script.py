import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from os.path import join as pjoin
import os


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = plt.subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                       trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                       color='blue')
    

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()


def get_frame(data, relative_frame_index):
    assert relative_frame_index >= 0
    assert relative_frame_index <= 1
    frame_number = data.shape[0]
    fnum = (int)((frame_number-1) * relative_frame_index)
    return data[fnum], fnum

def generate_sketches(motion_name, save_path, kinematic_tree, joints, radius = 2, relative_frame_indexes=None): 
    """This function can be used to generate sketches from a given motion datafile

    Args:
        motion_name (str): name of the motion passed to this function
        save_path (str): root directory path, in this directory a new directory named after the motion name will be created
        kinematic_tree (list): list of kinematic chains
        joints (np.array): data of the motion of the shape (t, 22, 3)
        radius (int, optional): Radius of the drawing. Defaults to 2.
        relative_frame_indexes (list, optional): Relative values of the frame location that should be extracted, all values must be between 0 and 1. Defaults to [0,0.5,1].
    """
    
    data = joints * -2.5
    data = data.reshape(len(data), -1, 3)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
   


    path_root = save_path
    if relative_frame_indexes is not None:
        for frame in relative_frame_indexes:
            save_path = path_root
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([radius/3, 4/3*radius ])
            ax.view_init(elev=160, azim=-90)
            ax.grid(b=False)
            ax.invert_zaxis()
            f, fnum = get_frame(data, frame)
            for c in kinematic_tree:
                points = f[c]
                ax.plot3D(points[:, 0], points[:, 2], points[:, 1], linestyle='-', c="black")

            # Show the plot
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            plt.axis('off')
            os.makedirs(save_path, exist_ok=True)
            print("SAVING " + motion_name)
            plt.savefig(pjoin(save_path, f"{motion_name}_{fnum}.png"))
            plt.clf()
        plt.close()
    else:
      
        indexes = list(range(data.shape[0]))[0::5]
        for i in indexes:
            frame = data[i]
            save_path = path_root
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([radius/3, 4/3*radius ])
            ax.view_init(elev=160, azim=-90)
            ax.grid(b=False)
            ax.invert_zaxis()
            for c in kinematic_tree:
                points = frame[c]
                ax.plot3D(points[:, 0], points[:, 2], points[:, 1], linestyle='-', c="black")

            # Show the plot
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            plt.axis('off')
            os.makedirs(save_path, exist_ok=True)
            #print("SAVING " + motion_name)
            plt.savefig(pjoin(save_path, f"{motion_name}_{i}.png"))
            plt.clf()
            plt.close()




if __name__ == "__main__":
    # Here we will create our sketch dataset of HumanAct12 dataset
    kinematic_tree =  [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21, 23], [9, 13, 16, 18, 20, 22]]

    import pickle as pkl

    with open("/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/humanact12poses.pkl", "rb") as f:
        data = pkl.load(f)

    data = data["joints3D"]
    
    import tqdm
    for i in tqdm.tqdm(range(len(data))):
        
        sample = data[i]
        save_path ="/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses/sketches"

        generate_sketches(f"{i}", save_path, kinematic_tree, sample)
        