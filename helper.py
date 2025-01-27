from PIL import Image
import os

import numpy as np

def save_tensor_as_image(tensor, directory = "/home/nkombol/Open Vocabulary Baseline/test_images", filename="image.png"):
    os.makedirs(directory, exist_ok=True)
    
    if tensor.max() > 1.0 or tensor.min() < 0.0:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).byte()  
    
    if tensor.ndimension() == 3 and tensor.shape[0] == 3: 
        image = tensor.permute(1, 2, 0).cpu().numpy()  
    elif tensor.ndimension() == 2:  
        image = tensor.cpu().numpy()
    else:
        raise ValueError("Tensor must have shape CxHxW (3 channels) or HxW (1 channel).")
    
    image = Image.fromarray(image)
    save_path = os.path.join(directory, filename)
    image.save(save_path)
    print(f"Image saved to {save_path}")


from PIL import Image
import numpy as np
import os

CITYSCAPES_COLORS = np.array([
    [128, 64, 128],  # Road
    [244, 35, 232],  # Sidewalk
    [70, 70, 70],    # Building
    [102, 102, 156], # Wall
    [190, 153, 153], # Fence
    [153, 153, 153], # Pole
    [250, 170, 30],  # Traffic light
    [220, 220, 0],   # Traffic sign
    [107, 142, 35],  # Vegetation
    [152, 251, 152], # Terrain
    [70, 130, 180],  # Sky
    [220, 20, 60],   # Person
    [255, 0, 0],     # Rider
    [0, 0, 142],     # Car
    [0, 0, 70],      # Truck
    [0, 60, 100],    # Bus
    [0, 80, 100],    # Train
    [0, 0, 230],     # Motorcycle
    [119, 11, 32],   # Bicycle
    [0, 0, 0]        # Unlabeled
], dtype=np.uint8)

def save_tensor_as_cityscapes_image(tensor, directory="/home/nkombol/Open Vocabulary Baseline/test_images", filename="image.png"):
    os.makedirs(directory, exist_ok=True)
    
    if tensor.ndimension() != 2:
        raise ValueError("Tensor must have shape HxW (class indices).")
    
    class_indices = tensor.cpu().numpy().astype(np.uint8)
    class_indices[class_indices == 255] = len(CITYSCAPES_COLORS) - 1
    color_image = CITYSCAPES_COLORS[class_indices]
    
    image = Image.fromarray(color_image)
    
    save_path = os.path.join(directory, filename)
    image.save(save_path)
    print(f"Image saved to {save_path}")

