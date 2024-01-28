import math  #
from os.path import join as pjoin
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import cv2
from textwrap import wrap
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from scipy.spatial.transform import Rotation as R

# import CLIP_image_encoder
model_name = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_name)
sketchEncoder = CLIPModel.from_pretrained(model_name)
sketchEncoder.eval()
for param in sketchEncoder.parameters():
    param.requires_grad = False


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


def get_frame(data, relative_frame_index):
    assert relative_frame_index >= 0
    assert relative_frame_index <= 1
    frame_number = data.shape[0]
    fnum = (int)((frame_number - 1) * relative_frame_index) + 1
    return data[fnum], fnum


def generate_vector_representation(motion_name, save_path, kinematic_tree, joints, relative_frame_indexes=[0, 0.5, 1]):
    data = joints * 1.5
    data = data.reshape(len(data), -1, 3)
    motion_length = joints.shape[0]
    relative_frame_indexes = []
    for i in range(motion_length):
        if i % 10 == 0:
            relative_frame_indexes.append(i)
    # relative_frame_indexes /= motion_length

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    path_root = save_path
    focal_length_x = 8  # Focal length in the x-direction
    focal_length_y = 8  # Focal length in the y-direction
    principal_point_x = 1  # Principal point x-coordinate
    principal_point_y = 1  # Principal point y-coordinate
    camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
                              [0, focal_length_y, principal_point_y],
                              [0, 0, 1]], dtype=np.float64)

    # Define the distortion coefficients
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)  # No distortion coefficients

    # Define the rotation and translation vectors
    rvec = np.zeros((3, 1), dtype=np.float64)  # Rotation vector
    tvec = np.array([[0], [0], [8]], dtype=np.float64)  # Translation vector

    for frame in relative_frame_indexes:
        save_path = path_root
        # f, fnum = get_frame(data, frame)
        f, fnum = data[frame], frame
        points = np.empty((0, 2))
        for c in kinematic_tree:
            dat = np.array(f[c], dtype=np.float64)
            dat = dat.reshape(-1, 3)[np.newaxis, ...]
            # Define the camera matrix

            # Project the 3D points to 2D
            image_points, _ = cv2.projectPoints(dat, rvec, tvec, camera_matrix, dist_coeffs)
            # pytorch_matrix

            chain_points = image_points[:, 0, :]
            i = len(chain_points) - 1
            while i - 1 >= 0:
                vec = chain_points[i] - chain_points[i - 1]
                i -= 1
                points = np.append(points, vec[np.newaxis, ...], axis=0)
        save_path = pjoin(save_path, motion_name)
        os.makedirs(save_path, exist_ok=True)
        save_path = pjoin(save_path, f"{motion_name}_{fnum}_vec.npy")
        np.save(save_path, points)


def generate_vector_dataset(dataset_path, dataset, save_directory, kinematic_tree):
    # Read the txt file containing the list of npy files
    with open(os.path.join(dataset_path, f"{dataset}.txt"), "r") as file:
        npy_files = file.read().splitlines()

    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Iterate over each npy file in the dataset
    for npy_file in npy_files:
        # npy_file = npy_file + '.npy'
        # Load the joints data from the npy file
        joints = np.load(os.path.join(dataset_path, "new_joints", npy_file + '.npy'))

        # Generate the vector representation for the given relative frames
        generate_sketches(npy_file, save_directory, kinematic_tree, joints)


def generate_sketches(motion_name, save_path, kinematic_tree, joints, radius=1.5, relative_frame_indexes=[0, 0.5, 1]):
    """This function can be used to generate sketches from a given motion datafile

    Args:
        motion_name (str): name of the motion passed to this function
        save_path (str): root directory path, in this directory a new directory named after the motion name will be created
        kinematic_tree (list): list of kinematic chains
        joints (np.array): data of the motion of the shape (t, 22, 3)
        radius (int, optional): Radius of the drawing. Defaults to 2.
        relative_frame_indexes (list, optional): Relative values of the frame location that should be extracted, all values must be between 0 and 1. Defaults to [0,0.5,1].
    """

    data = joints * 1.4
    data = data.reshape(len(data), -1, 3)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    # abs frames
    motion_length = joints.shape[0]
    relative_frame_indexes = []
    for i in range(motion_length):
        if i % 10 == 0 or i == motion_length - 1:
            relative_frame_indexes.append(i)

    camera_position = np.array([0, 0, 7])
    rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    angle_x, angle_y, angle_z = 20, 5, 180  # Angles in degrees
    rotation = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
    rotation_matrix = rotation.as_matrix()
    rot_mat = np.dot(rotation_matrix, rot_mat)
    path_root = save_path
    for frame in relative_frame_indexes:
        save_path = path_root
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim3d([-radius / 2, radius / 2])
        # ax.set_ylim3d([0, radius])
        # ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([radius * 1 / 3., radius * 4 / 3.])
        ax.view_init(elev=160, azim=90)
        ax.dist = 8.5
        ax.grid(b=False)
        ax.invert_zaxis()
        ax.invert_yaxis()
        # ax.invert_xaxis()
        # f, fnum = get_frame(data, frame)
        f, fnum = data[frame], frame
        colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
        colors = ['red', 'orange', 'pink', 'green', 'blue']
        for c, color in zip(kinematic_tree, colors):
            points = f[c]
            ax.plot3D(points[:, 0], points[:, 2], points[:, 1], linestyle='-', linewidth=4, color='black')
        # for i in range(f.shape[0]):
        #     ax.scatter(f[i, 0], f[i, 2], f[i, 1], color='black', s=5)

        # Show the plot
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.axis('off')
        """
        f, fnum = data[frame], frame
        f = np.dot(rot_mat, f.T).T
        f -= camera_position
        points2D = 3 * f[:, :2] / f[:, 2, None]
        colors = ['red', 'orange', 'pink', 'green', 'blue']
        for c, color in zip(kinematic_tree, colors):
            points = f[c]
            plt.plot(points2D[c, 0], points2D[c, 1], linewidth=4, color="black")
        plt.xlim(points2D[0, 0] - 0.6, points2D[0, 0] + 0.6)
        plt.ylim(points2D[0, 1] - 0.7, points2D[0, 1] + 0.5)
        plt.axis('off')
        save_path = pjoin(save_path, motion_name)
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        print()
        plt.savefig(pjoin(save_path, f"{motion_name}_{fnum}.png"))
        plt.clf()

        # embedded sketches
        img = Image.open(pjoin(save_path, f"{motion_name}_{fnum}.png"))
        img_emb = processor(images=img, return_tensors="pt", padding=True)
        img_emb = sketchEncoder.get_image_features(**img_emb)
        np.save(pjoin(save_path, f"{motion_name}_{fnum}.npy"), img_emb)




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
        data *= -1.5  # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
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
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

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


def main():
    datapath = r'F:\ADL\CV\s2m_with_joint_position_loss\test_data'
    savepath = r'F:\ADL\CV\s2m_with_joint_position_loss\test_data\sketches4'
    os.makedirs(savepath, exist_ok=True)
    generate_vector_dataset(dataset_path=datapath, dataset='train', save_directory=savepath,
                            kinematic_tree=[[0, 2, 5, 8, 11],
                                            [0, 1, 4, 7, 10],
                                            [0, 3, 6, 9, 12, 15],
                                            [9, 14, 17, 19, 21],
                                            [9, 13, 16, 18, 20]])
    pass


if __name__ == '__main__':
    main()