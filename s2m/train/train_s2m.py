from .data_loaders import 
import torch
def main():
    data_path = "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/tests/test_data/opt_test.txt"
    loader = dataloader.get_dataset_loader(datapath=data_path, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #model, diffusion = create_model_and_diffusion(args, data)