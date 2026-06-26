import numpy as np
from matplotlib import pyplot as plt
import os

# This gets the folder where your dome.py script is located
script_dir = os.path.dirname(__file__) 
# Join the folder path with the (corrected) filename
file_path = os.path.join(script_dir, 'images.npy') 

try:
    img_array = np.load(file_path)
    print("Array Shape:", img_array.shape)
    
    # Simple check: if it's a collection of images, pick the first one
    if img_array.ndim == 4: 
        plt.imshow(img_array[55])
    else:
        plt.imshow(img_array)
        
    plt.show()
    
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}")
    print("Check if the filename is 'images.npy' or 'iamges.npy'.")
