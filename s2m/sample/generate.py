from s2m.utils.parser_util import generate_args
import os
from s2m.model.mdm import MDM
from s2m.diffusion import Gaussian_diffusion as gd
from s2m.data_loaders.dataset import HumanML3D 
from s2m.data_loaders.humanml.scripts.motion_process import recover_from_ric
from s2m.model.rotation2xyz import Rotation2xyz
import s2m.data_loaders.humanml.utils.paramUtil as paramUtil
from s2m.data_loaders.humanml.utils.plot_script import plot_3d_motion
import torch as th
from PIL import Image
import numpy as np
from PIL.ImageOps import grayscale
from torchvision.transforms import GaussianBlur, RandomRotation, ToTensor, Compose

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        print("cuda")
        return th.device(f"cuda:0")
    return th.device("cpu")
def main():

    args = generate_args()
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    fps = 20
    assert args.input_sketch
    not_using_data = not args.input_sketch
    if not_using_data:
        sketch = None
    else:
        sketch = Image.open(args.input_sketch)
        sketch = sketch.crop((200, 200, 800, 800))
        sketch = sketch.resize((200,200))
        sketch = grayscale(sketch)

        transforms = Compose([
            # RandomRotation([-20, 20], fill=255),
            # GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
            ToTensor()
            ]
            )
        sketch = transforms(sketch)
        sketch = th.unsqueeze(sketch, 0)
   
    args.num_samples = 1
    args.batch_size = 1

    model = MDM()
    betas = gd.get_named_beta_schedule(schedule_name="cosine", num_diffusion_timesteps=1000)
    state_dict = th.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(dev())
    model.eval()

    diffusion = gd.GaussianDiffusion(betas=betas)
    max_frames = 86

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
            model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
            clip_denoised=False,
            sketch=sketch,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    print(sample.shape)
    loader = HumanML3D(datapath='/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/dataset/HumanML3D/opt_test.txt')
    n_joints = 22
    
    sample = loader.t2m_dataset.inverse_norm(sample.cpu().permute(0,2,3,1)).float()
    sample = recover_from_ric(sample, joints_num  = n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    rot2xyz_pose_rep = 'xyz'
    rot2xyz_mask = None
    rot2xyz = Rotation2xyz(device="cpu", dataset = HumanML3D)
    sample = rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                           jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                           get_rotations_back=False)
    sample = sample.cpu().numpy()
    out_path = args.output_dir
    if not out_path:
        out_path = os.path.join(os.path.dirname(args.model_path), 'generated_sample')
    os.makedirs(out_path, exist_ok=True)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, sample)

    print(f"saving visualizations to [{out_path}]...")
    animation_save_path = os.path.join(out_path, "animation.mp4")
    skeleton = paramUtil.t2m_kinematic_chain
    print(sample.shape)
    motion = sample[0, ...].transpose(2, 0, 1)
    plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title="", fps=fps)
