
import os 
from torch.utils import data 
import random
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
class EvalData(data.Dataset):
    def __init__(self):
        sketch_dir = "/home/xie/code/HumanMotionGeneration/test_data/sketches"
        train_file = "/home/xie/code/HumanMotionGeneration/test_data/train2.txt"
        train_paths = []
        with open(train_file, 'r') as file:
            for line in file:
                sample_name = line.strip()
                sample_path = os.path.join(sketch_dir, sample_name)
                train_paths.append(sample_path)
        png_files = {}

        for subdir in train_paths:
            png_files[subdir.split("/")[-1]] = [os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith(".png")]
        self.sketch_data = png_files
        self.names = list(png_files.keys())

    def __getitem__(self, index):
        sk1 = self.sketch_data[self.names[index]]
        sk1_images = []
        for sketch in sk1:
            image = np.where(np.array(Image.open(sketch))> 0, 1, 0)
            sk1_images.append(image)
        comp_index = random.randint(0, len(self.names)-1)
        sk2 = self.sketch_data[self.names[comp_index]]
        sk2_images = []
        for sketch in sk2:
            image = np.where(np.array(Image.open(sketch)) > 0, 1, 0)
            sk2_images.append(image)
      
        for s1, s2 in zip(sk1_images, sk2_images): 
           
            #print(np.max(np.sqrt(np.square(s1-s2))))
            diff = np.sum(np.mean(np.sqrt(np.square(s1-s2))))
        return (index, comp_index), 1-diff

    def __len__(self):
        return len(self.names)
my_dataset = EvalData()
loader = DataLoader(my_dataset, batch_size=1, shuffle=True)

data_points = []
for (i, ci), diff in loader:
    data_points.append(diff.item())
    if len(data_points) > 1000:
        break

import matplotlib.pyplot as plt 

 # Calculate mean and standard deviation
mean = np.mean(data_points)
std = np.std(data_points)
print(mean)
# Create a range of values for x-axis
x = np.linspace(mean - 3 * std, mean + 3 * std, 100)

# Calculate Gaussian probability for each x value
y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# Plot the Gaussian probability graph
plt.plot(x, y)
plt.xlabel('Data')
plt.ylabel('Probability')
plt.title('Probability of Sketch Similarities')
plt.savefig("test.png")
plt.show()