
import json
import os
import torch
import numpy as np
from regression_model.model import ResNetCNN
from regression_model.dataloader import transform_img
from PIL import Image
import utils.rotation_conversions as geometry

def preprocess_inputs(condition_path_dir):
    """
    Args: Condition_path_dir: str 
          The condition directory needs to contain the following files:
          - n image files of sketches (png format)
          - a json file with the following syntax named "condition.json":
            {
                "frames": [
                    {   
                        "file_name": str,
                        "relative_frame_id": int,
                        "translation": [float, float, float],
                    },
                    ...
                ]
                "regression_model_path": Path to the trained sketch regression model
            }
    """


    # Load the json 
    with open(os.path.join(condition_path_dir, "condition.json"), "r") as f:
        condition = json.load(f)
    
    # Load the torch ResNet18 regression model
    regression_model_path = condition["regression_model_path"]
    regression_model = ResNetCNN(ttype="poses")
    regression_model.load_state_dict(torch.load(regression_model_path, map_location=torch.device("cpu")))
    regression_model.eval()

    # Load the sketches
    sketches = []
    sketch_frames = []
    for frame in condition["frames"]:
        sketch_path = os.path.join(condition_path_dir, frame["file_name"])
        sketch = Image.open(sketch_path).convert("L")
        sketch = transform_img(sketch)
        sketches.append(sketch) 
        sketch_frames.append(frame["relative_frame_id"])
    # If CUDA is available, move the model to the GPU
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model.to(device)

    # Preprocess the sketches
    sketches = torch.stack(sketches).to(device)
    #sketches = sketches.unsqueeze(1)
    # Predict the poses
    poses = regression_model(sketches)
    # Convert the poses to numpy
    poses = poses

    # Load the sketch translations
    translations = []
    for frame in condition["frames"]:
        translations.append(frame["translation"])
    translations = np.array(translations)
    poses = poses.cpu()
    # Transform each pose to a rot6d representation 
    rot6d = []
    poses = poses.reshape(poses.shape[0], 24, 3)
    
    rot6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(poses))
        
    print(rot6d.shape)
    translations = torch.tensor(translations)
    padded_tr = torch.zeros((rot6d.shape[0], rot6d.shape[2]), dtype=rot6d.dtype)
    padded_tr[:, :3] = translations
    print(padded_tr.shape)
    ret = torch.cat((rot6d, padded_tr[:, None]), 1)
    ret = ret.permute(1,2,0).contiguous()
    print("ret", ret.shape)
    # Return the rot6d representation as well an array of relative frame indexes

    return {"cond": ret.float(), "frames" :sketch_frames}


if __name__ == "__main__":
    
    p = "/media/jan/SSD Spiele/ADLCV/HumanMotionGeneration/ucs2m/dataset/condtion_/m1"
    _, sketch_ids = preprocess_inputs(p)
    print(sketch_ids)