from data_loaders.humanml.networks.modules import *
from data_loaders import dataloader
from data_loaders.dataloader import get_dataset_loader
import torch
import scipy.linalg as linalg
from model.mdm import MDM
from diffusion import Gaussian_diffusion as gd
from tqdm import tqdm
from scipy.linalg import norm
from numpy.linalg import matrix_power
def calculate_diversity(activation, diversity_times):
    print(activation.shape)
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1).astype(float)
    mu2 = np.atleast_1d(mu2).astype(float)

    sigma1 = np.atleast_2d(sigma1).astype(float)
    sigma2 = np.atleast_2d(sigma2).astype(float)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    #test = sigma1.dot(sigma2)
    #print(test[0,0])
    #covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=True)
    #error = norm(covmean @ covmean - test, 'fro')
    #print(error)
    eigenvalues, eigenvectors = np.linalg.eig(sigma1.dot(sigma2))

    # Take the square root of the eigenvalues
    sqrt_eigenvalues = np.sqrt(eigenvalues)

    # Reconstruct the matrix with the square root of the eigenvalues
    sqrt_A = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)

    # Calculate the error
    error = np.linalg.norm(sqrt_A @ sqrt_A - sigma1.dot(sigma2), 'fro')
    print(f'Error: {error}')
    covmean = sqrt_A
   
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_motion_embeddings(motions, m_lens, encoder, model):
   
    with torch.no_grad():
        motions = motions.detach().to("cpu").float()
    
        #align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
     
        #motions = motions[align_idx]
        #m_lens = m_lens[align_idx]
    
        '''Movement Encoding'''
        movements = encoder(motions[..., :-4]).detach()
        m_lens = m_lens // 4
        motion_embedding = model(movements, m_lens)
    return motion_embedding


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


if __name__ == "__main__":

    path_to_eval_models = "./finest.tar"
    # We load the pretrained Motion Encoder
    checkpoint = torch.load(path_to_eval_models)
    encoder = MovementConvEncoder(input_size=259, hidden_size=512, output_size=512)
    encoder.load_state_dict(checkpoint['movement_encoder'])
    model = MotionEncoderBiGRUCo(input_size=512, hidden_size=1024, output_size=512, device="cpu")
    model.load_state_dict(checkpoint['motion_encoder'])

    encoder.eval()
    model.eval()
    """
    They wrap this one in n-repititions (different noise generations from the model)
    """

    """
    Code for the dataset loading goes here, 
    We need a gt (ground truth) loader that loads the test dataset.
    Either you already create the test motions from the trained model (best) or we do that here, 
    then we need to load the model and do inference here.


    They Call the Model in CompMDMGeneratedDataset (data_loaders/humanml/motion_loaders/comp_v6_model_dataset), 
    which is called from eval_humanml3d.py (in eval)
    by get_mdm_loader

    They iterate over the test split dataset and just create the motions by calling the sampling fn. 
    I guess you can just copy the adjusted code from the diffusion here (they just call sample_fn)

    I think it makes sense to wrap that in another simle dataset class that stores these embeddings, 
    a simple list would also do, then we need to adjust the gen_loader code though. 

    """
    # create model and diffusion
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = r'F:\ADL\CV\s2m_with_joint_position_loss\save\server_black2_0.5\trained_model_mN100_2.3.pth'
    print("Creating model and diffusion...")
    model_MDM = MDM()
    betas = gd.get_named_beta_schedule(schedule_name="cosine", num_diffusion_timesteps=1000)
    # loss_type = gd.LossType.MSE
    batch_size = 1
    gt_loader = get_dataset_loader(datapath='test_data/humanml_opt.txt', batch_size=batch_size, split='test')
    diffusion = gd.GaussianDiffusion(betas=betas, loader=gt_loader)
    print('loading pre-trained model: /home/xie/code/HumanMotionGeneration/save/mN100-10_BS1_3e-5_f0.8_p150_s7000k_black_no_fixed_length3/trained_model2.pth')
    state_dict = torch.load(r'/home/xie/code/HumanMotionGeneration/save/mN100-10_BS1_3e-5_f0.8_p150_s7000k_black_no_fixed_length3/trained_model2.pth', map_location='cpu')
    missing_keys, unexpected_keys = model_MDM.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    model_MDM.to(device)
    model_MDM.eval()

    """

    Result from here a gt_loader (ground truth)
                     a gen_loader (generated motions)
    """

    # FID EVALUATION


    gen_motion_embeddings = []
    for i, batch in tqdm(enumerate(gt_loader)):
        print(f"{i} / {len(gt_loader)}")  # here just use gt loader again
        """
        the output of dataloader is: (motion, sketch, key_frame)
        here we generate the motion from GT condition 
        """
        with torch.no_grad():
            motion_x, sketch, keyframe = batch
            sketch = sketch.to(device)
            keyframe = keyframe.to(device)
            sample_fn = diffusion.p_sample_loop
            max_frames = motion_x.shape[1]
            motions = sample_fn(
                model_MDM,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (batch_size, model_MDM.njoints, model_MDM.nfeats, max_frames),  # BUG FIX
                clip_denoised=False,
                sketch=sketch,
                keyframe=keyframe,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            # motions.shape: [batch_size, 263, 1, motionLength]
            m_lens = torch.tensor([motions.shape[3]])
            
            motions = motions.permute(0,2,3,1)[0,...]
            motion_embeddings = get_motion_embeddings(motions, m_lens, encoder, model).detach().cpu().numpy()
            gen_motion_embeddings.append(motion_embeddings)
        # if i > 305:
        #     break
    gen_motion_embeddings = np.concatenate(gen_motion_embeddings, axis=0)
    gen_mu, gen_cov = calculate_activation_statistics(gen_motion_embeddings)
    print(gen_mu)

    gt_motion_embeddings = []
    for i, batch in tqdm(enumerate(gt_loader)):
        with torch.no_grad():
            # _, _, _, sent_lens, motions, m_lens, _ = batch
            motions, _, _ = batch # motions.shape: [batch_size, motionLength, 263]
            m_lens = torch.tensor([motions.shape[1]])
            motion_embeddings = get_motion_embeddings(motions, m_lens, encoder, model).detach().cpu().numpy()
            gt_motion_embeddings.append(motion_embeddings)
        # if i > 305:
        #     break
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
    print(f'--->FID: {fid:.4f}')

    # DIVERSITY

    diversity_times = 300
    diversity = calculate_diversity(gen_motion_embeddings, diversity_times)
    print(f'--->Diversity: {diversity:.4f}')

    # The other metrics they use are related to text, so they dont make sense for us