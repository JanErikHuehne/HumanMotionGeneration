from data_loaders import dataloader
from model.mdm import MDM
from diffusion import Gaussian_diffusion as gd
import torch as th
import os
import json
from train.TrainLoop import *
from utils.fixseed import fixseed
from utils.parser_util import train_args, get_cond_mode

from utils import dist_util

def dev():
    """
    Get the device to use.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:0")
    return th.device("cpu")


def load_pretrained(model, model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(unexpected_keys) > 0:
        print(unexpected_keys)
    print(missing_keys)

def main():
    torch.cuda.empty_cache() 
    args = train_args()
    fixseed(args.seed)
    # train_platform_type = eval(args.train_platform_type)
    # train_platform = train_platform_type(args.save_dir)
    # train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    print("creating model and diffusion...")

    #arg_dict =get_model_args(args, loader)
    model = MDM()
    print(model.latent_dim)
    # load_pretrained(model, "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/pretrained_model/model000475000.pt")
    betas = gd.get_named_beta_schedule(schedule_name="cosine", num_diffusion_timesteps=1000)

    # # loading model
    # print('loading pre-trained model: /home/xie/code/HumanMotionGeneration/save/mN100_BS1_0.5e-5_f0.92_p200_s7000k_black/checkpoint2400000.pth')
    # state_dict = torch.load(r'/home/xie/code/HumanMotionGeneration/save/mN100_BS1_0.5e-5_f0.92_p200_s7000k_black/checkpoint2400000.pth', map_location='cpu')
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict['model_state_dict'], strict=False)
    # assert len(unexpected_keys) == 0

    # loading model
    print('loading pre-trained model: /home/xie/code/HumanMotionGeneration/save/mN100_BS1_5e-5_f0.8_p150_s7000k_black_no_fixed_length2/trained_model2.pth')
    state_dict = torch.load(r'/home/xie/code/HumanMotionGeneration/save/mN100_BS1_5e-5_f0.8_p150_s7000k_black_no_fixed_length2/trained_model2.pth', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0

    # loss_type = gd.LossType.MSE
    model.to(dev())
    print("creating data loader...")
    loader = get_dataset_loader(datapath='test_data/humanml_opt.txt', batch_size=args.batch_size, split='train2')
    val_loader = get_dataset_loader(datapath='test_data/humanml_opt.txt', batch_size=1, split='val')
    #loader = get_dataset_loader(datapath='/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/test_data/opt_test.txt', batch_size=1)
    # loader = get_dataset_loader(datapath='/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/dataset/HumanML3D/opt_test.txt', batch_size=64)
    # motion =
    diffusion = gd.GaussianDiffusion(betas=betas, loader=loader, val_loader=val_loader)

    print("Training...")
    TrainLoop(args, model, diffusion, data=loader, val_data=val_loader).run_loop()



def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}

if __name__ == '__main__':
    main()