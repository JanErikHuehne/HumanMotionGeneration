import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np
import torch as th
#import blobfile as bf
import torch
from torch.optim import AdamW
import s2m.diffusion.Gaussian_diffusion as gd
# from diffusion import logger
# from utils import dist_util
from .train_S2M import dev
# from diffusion.fp16_util import MixedPrecisionTrainer
# from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
# from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
# from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.dataloader import get_dataset_loader


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args,  model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        # self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        # self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.model_params = list(self.model.parameters())
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        # self._load_and_sync_parameters()
        # self.mp_trainer = MixedPrecisionTrainer(
        #     model=self.model,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=self.fp16_scale_growth,
        # )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite
        self.opt = AdamW(
            self.model_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # if self.resume_step:
            # self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = dev()
    
        self.schedule_sampler_type = 'uniform'
        # self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        # self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        # if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
        #     mm_num_samples = 0  # mm is super slow hence we won't run it during training
        #     mm_num_repeats = 0  # mm is super slow hence we won't run it during training
        #     gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                     split=args.eval_split,
        #                                     hml_mode='eval')
        #
        #     self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                            split=args.eval_split,
        #                                            hml_mode='gt')
        #     self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
        #     self.eval_data = {
        #         'test': lambda: eval_humanml.get_mdm_loader(
        #             model, diffusion, args.eval_batch_size,
        #             gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
        #             args.eval_num_samples, scale=1.,
        #         )
        #     }
        self.use_ddp = False
        self.ddp_model = self.model

    # def _load_and_sync_parameters(self):
    #     resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #
    #     if resume_checkpoint:
    #         self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
    #         logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
    #         self.model.load_state_dict(
    #             dist_util.load_state_dict(
    #                 resume_checkpoint, map_location=dist_util.dev()
    #             )
    #         )

    # def _load_optimizer_state(self):
    #     main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #     opt_checkpoint = bf.join(
    #         bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
    #     )
    #     if bf.exists(opt_checkpoint):
    #         logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
    #         state_dict = dist_util.load_state_dict(
    #             opt_checkpoint, map_location=dist_util.dev()
    #         )
    #         self.opt.load_state_dict(state_dict)

    def sample(self, batch_size):
        w = np.ones([self.log_interval])
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(dev())
        return indices

    def run_loop(self):
        self.num_epochs = 30000
        for epoch in range(self.num_epochs):
            loss = 0
           
            print(f'Starting epoch {epoch}')
            # for motion, sketch in tqdm(self.data):
            for motion, sketch in tqdm(self.data):

                motion = motion.to(self.device)
                sketch = sketch.to(self.device)

                loss += self.run_step(motion, sketch)
            
        
                # if self.step % self.log_interval == 0:
                #     for k,v in logger.get_current().name2val.items():
                #         if k == 'loss':
                #             print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))
                #
                #         if k in ['step', 'samples'] or '_q' in k:
                #             continue
                #         else:
                #             self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    # self.evaluate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
                break
            print("loss : {}".format(loss / 1 ))# (len(self.data))))
        ###
        # if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            # break
        ###
        # Save the last checkpoint if it wasn't already saved.
        if epoch % 100 == 0:
            self.save()
            # self.evaluate()
    def run_step(self, batch, sketch):
        loss = self.forward_backward(batch, sketch)
        self.opt.step()
        self._anneal_lr()
        return loss 
        # self.log_step()

    def forward_backward(self, batch, sketch):
        ##
        # zero_grad

        model_params = self.model_params
        for param in model_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_sketch = sketch
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t = self.sample(micro.shape[0])

            # compute_losses = functools.partial(
            #     self.diffusion.training_losses,
            #     self.ddp_model,
            #     micro,  # [bs, ch, image_size, image_size]
            #     t,  # [bs](int) sampled timesteps
            #     model_kwargs=micro_sketch,
            #     dataset=self.data
            # )


            losses = self.diffusion.training_losses(self.model, x_start=micro, sketch=micro_sketch, t=t)


            # if isinstance(self.schedule_sampler, LossAwareSampler):
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses["loss"].detach()
            #     )

            loss = (losses["loss"] ).mean()
            # log_loss_dict(
            #     self.diffusion, t, {k: v  for k, v in losses.items()}
            # )
           
            loss.backward()
            return loss

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    # def log_step(self):
    #     logger.logkv("step", self.step + self.resume_step)
    #     logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    # def save(self):
    #     def save_checkpoint(params):
    #         state_dict = self.mp_trainer.master_params_to_state_dict(params)
    #
    #         # Do not save CLIP weights
    #         clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
    #         for e in clip_weights:
    #             del state_dict[e]
    #
    #         logger.log(f"saving model...")
    #         filename = self.ckpt_file_name()
    #         with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
    #             torch.save(state_dict, f)
    #
    #     save_checkpoint(self.model.parameter)
    #
    #     with bf.BlobFile(
    #         bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
    #         "wb",
    #     ) as f:
    #         torch.save(self.opt.state_dict(), f)

    def save(self):
        model = self.model  # Replace MyModel with your model class
        # model.load_state_dict(torch.load('model_state_dict.pth'))
        torch.save(model, './save/trained_model1')
        torch.save(model.state_dict(), "./save/trained_model2")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


# def get_blob_logdir():
#     # You can change this to be a separate path to save checkpoints to
#     # a blobstore or some external drive.
#     return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


# def log_loss_dict(diffusion, ts, losses):
#     for key, values in losses.items():
#         logger.logkv_mean(key, values.mean().item())
#         # Log the quantiles (four quartiles, in particular).
#         for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
#             quartile = int(4 * sub_t / diffusion.num_timesteps)
#             logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
