import torch
import sys
import os
from pathlib import Path 

# Dynamically resolve path relative to this script's location
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'SAIS' / 'scripts' / 'dino-main'))  # Adjust if SAIS is in a different location
import vision_transformer as vits

def load_model(arch='vit_small', patch_size=16, weights_path=None, device='cpu'):
    """Load a DINO model."""
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)  # num_classes=0 for feature extraction   
    
    if weights_path and os.path.isfile(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        if 'student' in state_dict:
            state_dict = state_dict['student']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # Remove 'module.' if present
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}  # Remove 'backbone.' if present
        model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
    model.to(device)
    return model

# Load DINO
model_name = "vit_small"
model = load_model(model_name, weights_path='./SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth')  # Use CPU for test
print(f'✓ Loaded {model_name} successfully')

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Forward pass
with torch.no_grad():
    output = model(dummy_input)
    print(f'✓ Feature extraction works. Output shape: {output.shape}')




    
