
from s2m.model.mdm import MDM
from s2m.diffusion import Gaussian_diffusion as gd
from s2m.train.train_S2M import dev
import torch
from PIL import Image
from PIL.ImageOps import grayscale
from torchvision.transforms import GaussianBlur, RandomRotation, ToTensor, Compose

def single_sample(model_checkpoint, sample_path):


    print("Creating Model ...")
    model = MDM()
    betas = gd.get_named_beta_schedule(schedule_name="cosine", num_diffusion_timesteps=1000)
    diffusion = gd.GaussianDiffusion(betas=betas)
    print("Loading model checkpoint")
    state_dict = torch.load(model_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(dev())
    
    # Load sketch sample

    ssample = Image.open(sample_path)
    img = img.crop((200, 200, 800, 800))
    img = img.resize((200,200))
    img = grayscale(img)
    transforms = Compose([
            # RandomRotation([-20, 20], fill=255),
            # GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
            ToTensor()
            ]
            )
    img = transforms(img)
    img = img.to(dev())

    model.eval()
    