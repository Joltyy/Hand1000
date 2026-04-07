from omegaconf import OmegaConf
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.append('./stable-diffusion')
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

# ----------------------
# Args
# ----------------------
parser = argparse.ArgumentParser(description='Train fusion model on GPU (memory-optimized)')
parser.add_argument('--gesture', type=str, required=True, help='Gesture name')
parser.add_argument('--Lambda', type=float, default=0.7, help='Hyperparameter Lambda')
args = parser.parse_args()

# ----------------------
# Device
# ----------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------
# Load Model
# ----------------------
def load_model_from_config(config, ckpt, device=device):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    sd = torch.load(ckpt, map_location=device)["state_dict"]
    model = instantiate_from_config(config.model)
    model.to(device).half()  # half precision
    model.load_state_dict(sd, strict=False)
    model.eval()
    model.cond_stage_model.device = device
    torch.cuda.empty_cache()
    return model

# ----------------------
# Image utils
# ----------------------
def load_img(path, target_size=128):
    image = Image.open(path).convert("RGB")
    tform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor()
    ])
    image = tform(image)
    return 2.*image - 1.

def decode_to_im(model, samples, n_samples=1, nrow=1):
    with torch.no_grad():
        samples = model.decode_first_stage(samples.half())
    ims = torch.clamp((samples + 1) / 2, 0, 1)
    x_sample = 255. * rearrange(
        ims.cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c',
        n1=n_samples//nrow,
        n2=nrow
    )
    return Image.fromarray(x_sample.astype(np.uint8))

# ----------------------
# Fusion Model
# ----------------------
class FusionModel(nn.Module):
    def __init__(self, encode_size, feature_size):
        super().__init__()
        self.fc = nn.Linear(encode_size + feature_size, encode_size)

    def forward(self, encode, feature):
        feature = feature.unsqueeze(0)
        fused = torch.cat((encode, feature), dim=1)
        return self.fc(fused)

# ----------------------
# Read prompts & features
# ----------------------
def read_lines(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

image_folder = f"./{args.gesture}"
image_filenames_file = f"./{args.gesture}_BLIP2_file_name.txt"
prompts_file = f"./{args.gesture}_BLIP2_modified.txt"
feature_npy_path = f"./{args.gesture}.npy"

image_filenames = read_lines(image_filenames_file)
prompts = read_lines(prompts_file)

nums = len(image_filenames)

# ----------------------
# Model Setup
# ----------------------
config = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
ckpt = "./stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"

model = load_model_from_config(config, ckpt, device)
sampler = DDIMSampler(model)

fusion_model = FusionModel(encode_size=59136, feature_size=63).to(device).half()

# ----------------------
# Hyperparameters
# ----------------------
scale = 3
h = 128
w = 128
ddim_steps = 5
ddim_eta = 0.0
lr = 0.001
text_opt_steps = 3

# ----------------------
# Training Loop
# ----------------------
scaler = GradScaler()

for idx in tqdm(range(nums), desc="Processing images"):
    torch.cuda.empty_cache()
    
    # Load image
    image_path = os.path.join(image_folder, image_filenames[idx])
    image = load_img(image_path).unsqueeze(0).to(device).half()
    prompt = prompts[idx]

    # Encode
    with torch.no_grad():
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
        orig_embs = model.get_learned_conditioning([prompt])

    feature = torch.tensor(np.load(feature_npy_path), dtype=torch.float16).to(device)
    
    # ----------------------
    # Fusion
    # ----------------------
    fused_feature_orig = fusion_model(orig_embs.view(1, -1), feature)
    fused_feature_orig = fused_feature_orig.view(orig_embs.shape)
    
    fused_feature = (args.Lambda * orig_embs + (1 - args.Lambda) * fused_feature_orig).clone().detach().requires_grad_(True)

    # ----------------------
    # Optimize text embedding only
    # ----------------------
    opt = torch.optim.Adam([fused_feature], lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(text_opt_steps):
        opt.zero_grad()
        noise = torch.randn_like(init_latent).half()
        t_enc = torch.randint(1000, (1,), device=device)
        z = model.q_sample(init_latent, t_enc, noise=noise)
        with autocast():
            pred_noise = model.apply_model(z, t_enc, fused_feature)
            loss = loss_fn(pred_noise, noise)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    fused_feature.requires_grad = False
    torch.cuda.empty_cache()

# ----------------------
# Save Fusion Model Only
# ----------------------
torch.save({
    'fusion_model_state_dict': fusion_model.state_dict()
}, "./fusion_model_optimized_gpu.ckpt")