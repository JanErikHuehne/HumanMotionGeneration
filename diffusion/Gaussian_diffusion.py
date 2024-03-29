# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os
from scipy.spatial.transform import Rotation as R
import enum
import math
from data_loaders.humanml.utils.plot_script2 import generate_sketches
import numpy as np
import torch
import torch as th
from copy import deepcopy
# from diffusion.nn import mean_flat, sum_flat
# from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
from data_loaders.humanml.scripts import motion_process
from data_loaders.humanml.scripts.motion_process import recover_from_ric

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type=None,
        model_var_type=None,
        loss_type=None,
        rescale_timesteps=False,
        lambda_rcxyz=0.,
        lambda_vel=0.,
        lambda_pose=1.,
        lambda_orient=1.,
        lambda_loc=1.,
        data_rep='rot6d',
        lambda_root_vel=0.,
        lambda_vel_rcxyz=0.,
        lambda_fc=0.,
        loader=None,
        val_loader = None
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = LossType.MSE
        self.rescale_timesteps = rescale_timesteps
        self.data_rep = data_rep
        self.loader = loader
        self.val_loader = val_loader

        if data_rep != 'rot_vel' and lambda_pose != 1.:
            raise ValueError('lambda_pose is relevant only when training on velocities!')
        self.lambda_pose = lambda_pose
        self.lambda_orient = lambda_orient
        self.lambda_loc = lambda_loc

        self.lambda_rcxyz = lambda_rcxyz
        self.lambda_vel = lambda_vel
        self.lambda_root_vel = lambda_root_vel
        self.lambda_vel_rcxyz = lambda_vel_rcxyz
        self.lambda_fc = lambda_fc

        if self.lambda_rcxyz > 0. or self.lambda_vel > 0. or self.lambda_root_vel > 0. or \
                self.lambda_vel_rcxyz > 0. or self.lambda_fc > 0.:
            assert self.loss_type == LossType.MSE, 'Geometric losses are supported by MSE loss type only!'

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))



    def sum_flat(tensor):
        """
        Take the sum over all non-batch dimensions.
        """
        return tensor.sum(dim=list(range(1, len(tensor.shape))))


    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = self.sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = self.sum_flat(mask) * n_entries
        # print('mask', mask.shape)
        # print('non_zero_elements', non_zero_elements)
        # print('loss', loss)
        mse_loss_val = loss / non_zero_elements
        # print('mse_loss_val', mse_loss_val)
        return mse_loss_val


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, sketch=None, keyframe=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        x1 = torch.squeeze(x, dim=0)
        x1 = x1.permute((1, 2, 0))
        model_output = model(x1, sketch, keyframe, self._scale_timesteps(t), **model_kwargs)
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = model_output.permute((0, 3, 1, 2)) #[1, 263, 1, 82],[bs,n_joints, n_feats, frames]
        # if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
        #     inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
        #     assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        #     assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
        #     model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)
            # print('model_output', model_output.shape, model_output)
            # print('inpainting_mask', inpainting_mask.shape, inpainting_mask[0,0,0,:])
            # print('inpainted_motion', inpainted_motion.shape, inpainted_motion)

        # if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
        #     assert model_output.shape == (B, C * 2, *x.shape[2:])
        #     model_output, model_var_values = th.split(model_output, C, dim=1)
        #     if self.model_var_type == ModelVarType.LEARNED:
        #         model_log_variance = model_var_values
        #         model_variance = th.exp(model_log_variance)
        #     else:
        #         min_log = _extract_into_tensor(
        #             self.posterior_log_variance_clipped, t, x.shape
        #         )
        #         max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        #         # The model_var_values is [-1, 1] for [min_var, max_var].
        #         frac = (model_var_values + 1) / 2
        #         model_log_variance = frac * max_log + (1 - frac) * min_log
        #         model_variance = th.exp(model_log_variance)
        # else:
        model_variance, model_log_variance = self.posterior_variance, self.posterior_log_variance_clipped,
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            # print('model_variance', model_variance)
            # print('model_log_variance',model_log_variance)
            # print('self.posterior_variance', self.posterior_variance)
            # print('self.posterior_log_variance_clipped', self.posterior_log_variance_clipped)
            # print('self.model_var_type', self.model_var_type)

            # model_variance = _extract_into_tensor(model_variance, t, x.shape)
            # model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x


        pred_xstart = process_xstart(model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        sketch=None,
        keyframe=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            sketch=sketch,
            keyframe=keyframe,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(out["mean"])
        # print('const_noise', const_noise)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        # print('mean', out["mean"].shape, out["mean"])
        # print('log_variance', out["log_variance"].shape, out["log_variance"])
        # print('nonzero_mask', nonzero_mask.shape, nonzero_mask)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise # why we need to add noise again? for sampling
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        sketch=None,
        keyframe=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            # model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            sketch=sketch,
            keyframe=keyframe
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample
        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        sketch=None,
        keyframe=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        # if skip_timesteps and init_image is None:
        #     init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                sample_fn = self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    sketch=sketch,
                    keyframe=keyframe,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )
                yield out
                img = out["sample"]


    def training_losses(self, model, x_start, t, sketch=None, keyframe= None, model_kwargs=None, noise=None, dataset=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']
        enc = model
        # mask = model_kwargs['y']['mask']
        # get_xyz = lambda sample: enc.rot2xyz(sample, mask=None, pose_rep=enc.pose_rep, translation=enc.translation,
        #                                      glob=enc.glob,
        #                                      # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
        #                                      jointstype='smpl',  # 3.4 iter/sec
        #                                      vertstrans=False)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        terms["loss"] = 0
        model_output = model(x_t, sketch, keyframe, self._scale_timesteps(t), **model_kwargs)
        target = x_start
        assert target.shape == model_output.shape
        loss = self.l2_loss(target, model_output)  # mean_flat(rot_mse)
        loss = loss.sum(dim=list(range(1, len(loss.shape))))
        loss = loss / (model_output.shape[1] * model_output.shape[2])
        # camara_loss
        loss_xyz, loss_key_xyz, loss_camera, loss_key_camera = self.xyz_camera_loss(model_output=model_output, target=target, model=model)
        loss_xyz, loss_key_xyz, loss_camera, loss_key_camera = loss_xyz.sum(dim=list(range(1, len(loss_xyz.shape)))), \
            loss_key_xyz.sum(dim=list(range(1, len(loss_key_xyz.shape)))), \
            loss_camera.sum(dim=list(range(1, len(loss_camera.shape)))), \
            loss_key_camera.sum(dim=list(range(1, len(loss_key_camera.shape))))
        terms["loss"] += (0.4 * (0.1 * loss_xyz + 2 * loss_key_xyz + 0.5 * loss_camera + 20 * loss_key_camera) + 1.2 * loss)
        terms["loss_xyz"], terms["loss_key_xyz"], terms["loss_camera"], terms["loss_key_camera"], terms["loss_reconstruct"] = \
            loss_xyz, loss_key_xyz, loss_camera, loss_key_camera, loss
        # # camara_loss
        # n_joints = 22
        # model_output = torch.unsqueeze(model_output, dim=0)
        # model_output = model_output.permute((0, 3, 1, 2))
        # # sample = self.loader.dataset.t2m_dataset.inverse_norm(model_output.permute(0, 2, 3, 1)).float()
        # sample = model_output.permute(0, 2, 3, 1) * torch.tensor(self.loader.dataset.t2m_dataset.std).to(model_output.device) + torch.tensor(self.loader.dataset.t2m_dataset.mean).to(model_output.device)
        # sample = sample.float()
        # sample = recover_from_ric(sample, n_joints)
        # sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        #
        # rot2xyz_pose_rep = 'xyz'
        # rot2xyz_mask = None
        # sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
        #                        jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
        #                        get_rotations_back=False)
        # key_sample = sample[..., [0, 10, 20, 30, 40]]
        # target = torch.unsqueeze(target, dim=0)
        # key_target = target * torch.tensor(self.loader.dataset.t2m_dataset.std).to(target.device) + torch.tensor(self.loader.dataset.t2m_dataset.mean).to(target.device)
        # key_target = key_target.float()
        # key_target = recover_from_ric(key_target, n_joints)
        # key_target = key_target.view(-1, *key_target.shape[2:]).permute(0, 2, 3, 1)
        # key_target = key_target[..., [0, 10, 20, 30, 40]]  # can be changed with joints.npy

        return terms

    def xyz_camera_loss(self, model_output=None, target=None, model=None):
        n_joints = 22
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = model_output.permute((0, 3, 1, 2))
        # sample = self.loader.dataset.t2m_dataset.inverse_norm(model_output.permute(0, 2, 3, 1)).float()
        sample = model_output.permute(0, 2, 3, 1) * torch.tensor(self.loader.dataset.t2m_dataset.std).to(
            model_output.device) + torch.tensor(self.loader.dataset.t2m_dataset.mean).to(model_output.device)
        sample = sample.float()
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz'
        rot2xyz_mask = None
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        key_sample = sample[..., [0, 10, 20, 30, 40]]
        target = torch.unsqueeze(target, dim=0)
        target = target * torch.tensor(self.loader.dataset.t2m_dataset.std).to(target.device) + torch.tensor(
            self.loader.dataset.t2m_dataset.mean).to(target.device)
        target = target.float()
        target = recover_from_ric(target, n_joints)
        target = target.view(-1, *target.shape[2:]).permute(0, 2, 3, 1)
        key_target = target[..., [0, 10, 20, 30, 40]]
        ####
        # target_test = torch.squeeze(target).permute(2, 0, 1).cpu().numpy().copy()
        # generate_sketches('1', r'F:\ADL\CV\s2m_with_joint_position_loss\save', [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]], target_test, radius=1.4)
        ####
        loss1 = self.l2_loss(target, sample) / (sample.shape[1] * sample.shape[3])
        loss2 = self.l2_loss(key_target, key_sample) / (key_sample.shape[1] * key_sample.shape[3])
        loss3 = self.camera_loss(sample=sample, target=target)
        loss4 = loss3[:, [0, 10, 20, 30, 40], ...]

        return loss1, loss2, loss3, loss4

    def camera_loss(self, sample=None, target=None):
        camera_position = torch.tensor([0, 0, 7]).to(sample.device)
        rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        angle_x, angle_y, angle_z = 20, 5, 180  # Angles in degrees
        rotation = R.from_euler('xyz', [angle_x, angle_y, angle_z], degrees=True)
        rotation_matrix = rotation.as_matrix()
        rot_mat = np.dot(rotation_matrix, rot_mat)
        rot_mat = torch.tensor(rot_mat).to(sample.device).float()
        f = sample.permute(2, 1, 0, 3)
        f_reshaped = f.reshape(3, -1)  # Reshapes f to a shape of (3, 22*1*41)
        f_transformed = torch.matmul(rot_mat, f_reshaped)
        f_transformed = f_transformed.reshape(f.shape)
        f = f_transformed.permute(2, 3, 1, 0)
        f -= camera_position
        points2D = 3 * f[..., :2] / f[..., 2, None]
        f_target = target.permute(2, 1, 0, 3)
        f_target_reshaped = f_target.reshape(3, -1)
        f_target_transformed = torch.matmul(rot_mat, f_target_reshaped)
        f_target_transformed = f_target_transformed.reshape(f_target.shape)
        f_target = f_target_transformed.permute(2, 3, 1, 0)
        f_target -= camera_position
        points2D_target = 3 * f_target[..., :2] / f_target[..., 2, None]



        """
        visualization
        
        for i in range(41):
            points2D_target_np = torch.squeeze(points2D_target).cpu().numpy()[i]
            points2D_target_np *= 1.4
            colors = ['red', 'orange', 'pink', 'green', 'blue']
            for c, color in zip([[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]], colors):
                plt.plot(points2D_target_np[c, 0], points2D_target_np[c, 1], linewidth=4, color="black")
            plt.xlim(points2D_target_np[0, 0] - 0.6, points2D_target_np[0, 0] + 0.6)
            plt.ylim(points2D_target_np[0, 1] - 0.7, points2D_target_np[0, 1] + 0.5)
            plt.axis('off')
            save_path = r'F:\ADL\CV\s2m_with_joint_position_loss\save\3'
            os.makedirs(save_path, exist_ok=True)
            print(save_path)
            plt.savefig(pjoin(save_path, f"{i}.png"))
            plt.clf()
        

        return
        """
        return self.l2_loss(points2D, points2D_target) / (points2D.shape[1] * points2D.shape[2])


    def fc_loss_rot_repr(self, gt_xyz, pred_xyz, mask):
        def to_np_cpu(x):
            return x.detach().cpu().numpy()
        """
        pose_xyz: SMPL batch tensor of shape: [BatchSize, 24, 3, Frames]
        """
        # 'L_Ankle',  # 7, 'R_Ankle',  # 8 , 'L_Foot',  # 10, 'R_Foot',  # 11

        l_ankle_idx, r_ankle_idx = 7, 8
        l_foot_idx, r_foot_idx = 10, 11
        """ Contact calculated by 'Kfir Method' Commented code)"""
        # contact_signal = torch.zeros((pose_xyz.shape[0], pose_xyz.shape[3], 2), device=pose_xyz.device) # [BatchSize, Frames, 2]
        # left_xyz = 0.5 * (pose_xyz[:, l_ankle_idx, :, :] + pose_xyz[:, l_foot_idx, :, :]) # [BatchSize, 3, Frames]
        # right_xyz = 0.5 * (pose_xyz[:, r_ankle_idx, :, :] + pose_xyz[:, r_foot_idx, :, :])
        # left_z, right_z = left_xyz[:, 2, :], right_xyz[:, 2, :] # [BatchSize, Frames]
        # left_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)  # [BatchSize, Frames]
        # right_velocity = torch.linalg.norm(left_xyz[:, :, 2:] - left_xyz[:, :, :-2], axis=1)
        #
        # left_z_mask = left_z <= torch.mean(torch.sort(left_z)[0][:, :left_z.shape[1] // 5], axis=-1)
        # left_z_mask = torch.stack([left_z_mask, left_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # left_z_mask[:, :, 1] = False  # Blank right side
        # contact_signal[left_z_mask] = 0.4
        #
        # right_z_mask = right_z <= torch.mean(torch.sort(right_z)[0][:, :right_z.shape[1] // 5], axis=-1)
        # right_z_mask = torch.stack([right_z_mask, right_z_mask], dim=-1) # [BatchSize, Frames, 2]
        # right_z_mask[:, :, 0] = False  # Blank left side
        # contact_signal[right_z_mask] = 0.4
        # contact_signal[left_z <= (torch.mean(torch.sort(left_z)[:left_z.shape[0] // 5]) + 20), 0] = 1
        # contact_signal[right_z <= (torch.mean(torch.sort(right_z)[:right_z.shape[0] // 5]) + 20), 1] = 1

        # plt.plot(to_np_cpu(left_z[0]), label='left_z')
        # plt.plot(to_np_cpu(left_velocity[0]), label='left_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 0]), label='left_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.plot(to_np_cpu(right_z[0]), label='right_z')
        # plt.plot(to_np_cpu(right_velocity[0]), label='right_velocity')
        # plt.plot(to_np_cpu(contact_signal[0, :, 1]), label='right_fc')
        # plt.grid()
        # plt.legend()
        # plt.show()

        gt_joint_xyz = gt_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        fc_mask = (gt_joint_vel <= 0.01)
        pred_joint_xyz = pred_xyz[:, [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx], :, :]  # [BatchSize, 4, 3, Frames]
        pred_joint_vel = torch.linalg.norm(pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
        pred_joint_vel[~fc_mask] = 0  # Blank non-contact velocities frames. [BS,4,FRAMES]
        pred_joint_vel = torch.unsqueeze(pred_joint_vel, dim=2)

        """DEBUG CODE"""
        # print(f'mask: {mask.shape}')
        # print(f'pred_joint_vel: {pred_joint_vel.shape}')
        # plt.title(f'Joint: {joint_idx}')
        # plt.plot(to_np_cpu(gt_joint_vel[0]), label='velocity')
        # plt.plot(to_np_cpu(fc_mask[0]), label='fc')
        # plt.grid()
        # plt.legend()
        # plt.show()
        return self.masked_l2(pred_joint_vel, torch.zeros(pred_joint_vel.shape, device=pred_joint_vel.device),
                              mask[:, :, :, 1:])
    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def foot_contact_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259
        contact_mask_gt = contact > 0.5  # contact mask order for indexes are fid_l [7, 10], fid_r [8, 11]
        vel_lf_7 = local_velocity[:, 7 * 3:8 * 3, :, :]
        vel_rf_8 = local_velocity[:, 8 * 3:9 * 3, :, :]
        vel_lf_10 = local_velocity[:, 10 * 3:11 * 3, :, :]
        vel_rf_11 = local_velocity[:, 11 * 3:12 * 3, :, :]

        calc_vel_lf_7 = ric_data[:, 6 * 3:7 * 3, :, 1:] - ric_data[:, 6 * 3:7 * 3, :, :-1]
        calc_vel_rf_8 = ric_data[:, 7 * 3:8 * 3, :, 1:] - ric_data[:, 7 * 3:8 * 3, :, :-1]
        calc_vel_lf_10 = ric_data[:, 9 * 3:10 * 3, :, 1:] - ric_data[:, 9 * 3:10 * 3, :, :-1]
        calc_vel_rf_11 = ric_data[:, 10 * 3:11 * 3, :, 1:] - ric_data[:, 10 * 3:11 * 3, :, :-1]

        # vel_foots = torch.stack([vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11], dim=1)
        for chosen_vel_foot_calc, chosen_vel_foot, joint_idx, contact_mask_idx in zip(
                [calc_vel_lf_7, calc_vel_rf_8, calc_vel_lf_10, calc_vel_rf_11],
                [vel_lf_7, vel_lf_10, vel_rf_8, vel_rf_11],
                [7, 10, 8, 11],
                [0, 1, 2, 3]):
            tmp_mask_gt = contact_mask_gt[:, contact_mask_idx, :, :].cpu().detach().numpy().reshape(-1).astype(int)
            chosen_vel_norm = np.linalg.norm(chosen_vel_foot.cpu().detach().numpy().reshape((3, -1)), axis=0)
            chosen_vel_calc_norm = np.linalg.norm(chosen_vel_foot_calc.cpu().detach().numpy().reshape((3, -1)),
                                                  axis=0)

            print(tmp_mask_gt.shape)
            print(chosen_vel_foot.shape)
            print(chosen_vel_calc_norm.shape)
            import matplotlib.pyplot as plt
            plt.plot(tmp_mask_gt, label='FC mask')
            plt.plot(chosen_vel_norm, label='Vel. XYZ norm (from vector)')
            plt.plot(chosen_vel_calc_norm, label='Vel. XYZ norm (calculated diff XYZ)')

            plt.title(f'FC idx {contact_mask_idx}, Joint Index {joint_idx}')
            plt.legend()
            plt.show()
        # print(vel_foots.shape)
        return 0
    # TODO - NOT USED YET, JUST COMMITING TO NOT DELETE THIS AND KEEP INITIAL IMPLEMENTATION, NOT DONE!
    def velocity_consistency_loss_humanml3d(self, target, model_output):
        # root_rot_velocity (B, seq_len, 1)
        # root_linear_velocity (B, seq_len, 2)
        # root_y (B, seq_len, 1)
        # ric_data (B, seq_len, (joint_num - 1)*3) , XYZ
        # rot_data (B, seq_len, (joint_num - 1)*6) , 6D
        # local_velocity (B, seq_len, joint_num*3) , XYZ
        # foot contact (B, seq_len, 4) ,

        target_fc = target[:, -4:, :, :]
        root_rot_velocity = target[:, :1, :, :]
        root_linear_velocity = target[:, 1:3, :, :]
        root_y = target[:, 3:4, :, :]
        ric_data = target[:, 4:67, :, :]  # 4+(3*21)=67
        rot_data = target[:, 67:193, :, :]  # 67+(6*21)=193
        local_velocity = target[:, 193:259, :, :]  # 193+(3*22)=259
        contact = target[:, 259:, :, :]  # 193+(3*22)=259

        calc_vel_from_xyz = ric_data[:, :, :, 1:] - ric_data[:, :, :, :-1]
        velocity_from_vector = local_velocity[:, 3:, :, 1:]  # Slicing out root
        r_rot_quat, r_pos = motion_process.recover_root_rot_pos(target.permute(0, 2, 3, 1).type(th.FloatTensor))
        print(f'r_rot_quat: {r_rot_quat.shape}')
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape}')
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 2, 3, 1)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21, 3)).type(th.FloatTensor)
        r_rot_quat_adapted = r_rot_quat[..., :-1, None, :].repeat((1,1,1,21,1)).to(calc_vel_from_xyz.device)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')
        print(f'r_rot_quat_adapted: {r_rot_quat_adapted.shape}, {r_rot_quat_adapted.device}')

        calc_vel_from_xyz = motion_process.qrot(r_rot_quat_adapted, calc_vel_from_xyz)
        calc_vel_from_xyz = calc_vel_from_xyz.reshape((1, 1, -1, 21 * 3))
        calc_vel_from_xyz = calc_vel_from_xyz.permute(0, 3, 1, 2)
        print(f'calc_vel_from_xyz: {calc_vel_from_xyz.shape} , {calc_vel_from_xyz.device}')

        import matplotlib.pyplot as plt
        for i in range(21):
            plt.plot(np.linalg.norm(calc_vel_from_xyz[:,i*3:(i+1)*3,:,:].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Calc Vel')
            plt.plot(np.linalg.norm(velocity_from_vector[:,i*3:(i+1)*3,:,:].cpu().detach().numpy().reshape((3, -1)), axis=0), label='Vector Vel')
            plt.title(f'Joint idx: {i}')
            plt.legend()
            plt.show()
        print(calc_vel_from_xyz.shape)
        print(velocity_from_vector.shape)
        diff = calc_vel_from_xyz-velocity_from_vector
        print(np.linalg.norm(diff.cpu().detach().numpy().reshape((63, -1)), axis=0))

        return 0





def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)