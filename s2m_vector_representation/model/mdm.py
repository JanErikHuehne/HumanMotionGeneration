import numpy as np
import torch
import torch.nn as nn
# import torch.nn.fun
from transformers import CLIPProcessor, CLIPModel
import clip
from model.rotation2xyz import Rotation2xyz

class MDM(nn.Module):

    def __init__(self, nfeats=263, latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", dataset='humanml3d', clip_dim=512):
            super().__init__()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.num_heads = num_heads
            self.latent_dim = latent_dim
            self.ff_size = ff_size
            self.dropout = dropout
            self.dataset = dataset
            self.activation = activation
            self.num_layers = num_layers
            self.input_feats = 263
            self.njoints = 263
            self.nfeats = 1
            self.input_process = InputProcess(self.input_feats, self.latent_dim)
            self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints,
                                                self.nfeats)
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
            self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation
                                                              )
            
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

            self.embded_timestep = TimestepEmbedder(latent_dim=self.latent_dim, sequence_pos_encoder=self.sequence_pos_encoder)
            #self.sketchEncoder = Sketch_Embedder(input_dim=200, latent_dim=self.latent_dim)
            model_name = "openai/clip-vit-base-patch16"
            self.sketchEncoder = CLIPModel.from_pretrained(model_name)
            self.sketchEncoder.eval()
            for param in self.sketchEncoder.parameters():
                param.requires_grad = False
            #######

            #######vector_encoder
            self.vectorEncoder = nn.Sequential(
                nn.Linear(21 * 2, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            #######
            self.linear_layer = nn.Linear(clip_dim, latent_dim)
            self.linear_layer2 = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.vectorReconstructor = nn.Sequential(
                nn.Linear(nfeats, 128),
                nn.SiLU(),
                nn.Linear(128, 21 * 2),
            )
            #reconstruction_of_vector
            # self.vector_reconstructor = nn.Sequential(
            #     nn.Linear(nfeats, ),
            #     nn.SiLU(),
            #     nn.Linear(self.latent_dim, self.latent_dim),
            # )
                     

    def forward(self, x, y, key_frames,timesteps):
        """
        x: [batch_size, n_feats, frames] x_t in the MDM paper
        y: [batch_size, 1, 200, 200] Sketch ImageInput
        timesteps: [batch_size] (int)
        """

        x = x.to(self.device) #x: [batch_size, n_feats, frames] x_t in the MDM paper
        y = y.to(self.device) #y: Sketch ImageInput
        timesteps = timesteps.to(self.device)
        timesteps_cache = timesteps
        time_emb = self.embed_timestep(timesteps) # yields [1, bs, d]
        time_emb1 =time_emb
        #######
        time_emb = time_emb.repeat(5, 1, 1) # yields [5, bs, d]
        #######
        #print("TIMESTEP EMBEDDING", emb.shape)
        ######
        # y['pixel_values'] = y['pixel_values'].reshape(-1, 3, 224, 224)
        y = y.reshape(y.shape[0], y.shape[1], -1)
        y = y.to(torch.float32)
        sketches_emb = self.vectorEncoder(y) #CLIP image encoder
        #######
        # sketches_emb = self.linear_layer(sketches_emb)
        ########
        sketches_emb = sketches_emb.reshape(-1, 5, self.latent_dim) # [bs, 5(sketches), d]
        sketches_emb = sketches_emb.permute(1, 0, 2) #[5(sketches), bs, d]
        ######
        frames_emb = torch.squeeze(self.sequence_pos_encoder.pe[key_frames[:, 0]], dim=1)
        frames_emb = torch.stack((frames_emb, torch.squeeze(self.sequence_pos_encoder.pe[key_frames[:, 1]], dim=1)), dim=0)
        frames_emb = torch.cat((frames_emb, self.sequence_pos_encoder.pe[key_frames[:, 2]].permute(1, 0, 2)), dim=0)
        frames_emb = torch.cat((frames_emb, self.sequence_pos_encoder.pe[key_frames[:, 3]].permute(1, 0, 2)), dim=0)
        frames_emb = torch.cat((frames_emb, self.sequence_pos_encoder.pe[key_frames[:, 4]].permute(1, 0, 2)), dim=0)
        frames_emb = self.linear_layer2(frames_emb)
        ######
        # HERE WE ADD THE ENCODING OF OUR 2D-SKETCHES
        emb =  sketches_emb + time_emb + frames_emb
        emb1 = time_emb1

        x = self.input_process(x)
        xseq = torch.cat((emb1, x), axis=0)  # [seqlen+5, bs, d]
        xseq = self.sequence_pos_encoder(xseq) # [seqlen+5, bs, d]
        output = self.seqTransEncoder(xseq)[1:] # [seqlen, bs, d]
        output = self.output_process(output)
        output_x = output.permute(1, 0, 2)
        selected_output = output_x[[0, 10, 20, 30, 40]].permute(1, 0, 2)

        predict_vectors = self.vectorReconstructor(selected_output)


        #output = self.output_process(output) # [bs, n_joints, nfeas, n_frames]
        return output, predict_vectors

class Sketch_Embedder(nn.Module):
    def __init__(self, input_dim=200, latent_dim=256):
        super(Sketch_Embedder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sketch_embed = nn.Sequential(
                            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.MaxPool2d(2,2),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                            nn.ReLU(),
                            nn.MaxPool2d(2,2),
                            nn.Flatten(),
                            nn.Linear(in_features=256, out_features=self.latent_dim)
                            )
        
    
    def forward(self, x):
        self.input_dimension = x.shape[1]
        res = self.sketch_embed(x)
        res = torch.reshape(res, shape=(1, res.shape[0], res.shape[1]))
        #print("CNN FORWARD SHAPE", res.shape)
        return res 
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    """
    This class represents the linear layer that mapps the joint motion distribution 
    into the expected latent dimension of the transformer model 
    """
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)


    def forward(self, x):
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x
      


class OutputProcess(nn.Module):
    """
    This class represents the linear mapping from the transformer output back into 
    the pose joint representation 
    """
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape

        # if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        # elif self.data_rep == 'rot_vel':
        #     first_pose = output[[0]]  # [1, bs, d]
        #     first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
        #     vel = output[1:]  # [seqlen-1, bs, d]
        #     vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
        #     output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        # else:
        #     raise ValueError
        output = output.reshape(bs, nframes, self.njoints)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
