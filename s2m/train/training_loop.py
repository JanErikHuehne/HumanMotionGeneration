import torch
from torch.optim import AdamW
from tqdm import tqdm


class TrainLoop():

    def __init__(self, args, model, diffusion, data):
        self.args = args
        self.data = data
        self.model = model
        self.diffusion = diffusion
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def run_loop(self):
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):

                motion = motion.to(self.device)
                cond = cond.to(self.device)


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
    

    def forward_backward(self, batch, cond):
        pass

