import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import create_dataloader
from model import ResNetCNN, VGGCNN
import tqdm
import numpy as np
from eval import generate_sketch, reverse_vector

JOINT_LENGTHS = np.array([0.11,
                 0.36,
                 0.38,
                 0.14,
                 0.11,
                 0.36,
                 0.38,
                 0.13,
                 0.12,
                 0.13,
                 0.05,
                 0.22,
                 0.1,
                 0.14,
                 0.12,
                 0.25,
                 0.25,
                 0.09,
                 0.14,
                 0.12,
                 0.25,
                 0.25,
                 0.09], dtype=float)[..., np.newaxis]

def train_model(dataset_path, num_epochs, batch_size, learning_rate, ttype="poses"):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.zeros(1).cuda()
    # Load the dataset
    dataloader = create_dataloader(dataset_path, batch_size, mode="train", ttype=ttype)
    test_loader = create_dataloader(dataset_path, batch_size, mode="test", ttype=ttype)
    # Define your model and move it to the device
    model = ResNetCNN(ttype=ttype).to(device)
    print(ResNetCNN(ttype=ttype))
    #print(model)
    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='none')  # Use reduction='none' for feature-wise loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.9, min_lr=1e-7)
    # Training loop
   
    for epoch in range(num_epochs):
        
         

      

        train_loss = 0.0
        for inputs, labels, _ in tqdm.tqdm(dataloader):
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            if ttype == "joints":
                labels = labels.reshape(-1, 23*3)
            elif ttype == "poses":
                labels = labels.reshape(-1, 24*3)
            # Forward pass
           
            outputs = model(inputs)
          
            loss = criterion(outputs, labels)

            # Compute feature-wise loss
            feature_loss = torch.mean(loss)
            train_loss += feature_loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            feature_loss.backward()
            optimizer.step()
        scheduler.step(feature_loss)
        model.eval()
        test_loss = 0
        for inputs, labels, _ in tqdm.tqdm(test_loader):
          
            inputs = inputs.to(device)
            labels = labels.to(device)
            if ttype == "joints":
                labels = labels.reshape(-1, 23*3)
            else:
                labels = labels.reshape(-1, 24*3)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = torch.mean(loss)
            test_loss += loss.item()
        model.train()
       
        # Update learning rate
        if epoch % 10 == 0:
              torch.save(model.state_dict(), f'{epoch}_trained_reg_{ttype}.pth')
        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train-Loss: {train_loss / len(dataloader):.4f}, Test-Loss: {test_loss / len(test_loader):.4f}")

        # Now we eval the loss
      

    # Save the trained model
    torch.save(model.state_dict(), f'final_trained_reg_{ttype}.pth')




if __name__ == "__main__":
    path = "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/HumanAct12Poses"
    train_model(path, 200, 16, 0.001)