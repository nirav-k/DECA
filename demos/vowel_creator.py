import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
identity_image = "TestSamples/examples/Mit/Mit cropped.jpg"
expression_images = [
    "TestSamples/exp/Face_A.png",
    "TestSamples/exp/Face_E.png",
    "TestSamples/exp/Face_I.png",
    "TestSamples/exp/Face_O.png",
    "TestSamples/exp/Face_U.png"
]
output_dir = "output_objs"
os.makedirs(output_dir, exist_ok=True)

# Initialize DECA
deca_cfg.model.use_tex = False
deca_cfg.rasterizer_type = 'pytorch3d'
deca = DECA(config=deca_cfg, device=device)

# Load identity image
testdata = datasets.TestData(identity_image, iscrop=True, face_detector='fan')
identity_data = testdata[0]
identity_image = identity_data['image'].to(device)[None,...]

# Encode identity
with torch.no_grad():
    id_codedict = deca.encode(identity_image)

#Save fully neutral exp image
neutral_codedict = id_codedict.copy()
neutral_codedict['exp'] = torch.zeros_like(neutral_codedict['exp'])  # Zero out expression
neutral_codedict['pose'][:, :3] = 0  # Frontal rotation
neutral_codedict['pose'][:, 3:] = 0  # Uncomment to center position

neutral_opdict, _ = deca.decode(neutral_codedict)
deca.save_obj(os.path.join(output_dir, "expression_0.obj"), neutral_opdict)
print("Saved neutral face: expression_0.obj")

for i, exp_img_path in enumerate(expression_images):
    # Load expression image
    expdata = datasets.TestData(exp_img_path, iscrop=True, face_detector='fan')
    exp_data = expdata[0]
    exp_image = exp_data['image'].to(device)[None,...]
    
    # Encode expression
    with torch.no_grad():
        exp_codedict = deca.encode(exp_image)
    
    # Transfer expression while keeping identity
    id_codedict['exp'] = exp_codedict['exp']  # Transfer expression
    
    # Reset head rotation (keep jaw movement from expression)
    id_codedict['pose'][:, :3] = 0  # Zero rotation (frontal)
    
    # Generate mesh
    transfer_opdict, _ = deca.decode(id_codedict)
    
    # Save OBJ file
    obj_name = os.path.join(output_dir, f"expression_{i+1}.obj")
    deca.save_obj_no_tex(obj_name, transfer_opdict)

    print(f"Saved: {obj_name}")

print("All expressions processed!")