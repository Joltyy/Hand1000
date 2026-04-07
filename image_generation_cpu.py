from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
from torch import nn
import argparse
import sys

sys.path.append('./stable-diffusion')

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

# ----------------------
# Arguments
# ----------------------
parser = argparse.ArgumentParser(description='Image Generation (CPU)')
parser.add_argument('--image_filenames_file', type=str, required=True)
parser.add_argument('--prompts_file', type=str, required=True)
parser.add_argument('--feature_npy_path', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--img_save_path', type=str, required=True)
parser.add_argument('--mu', type=float, required=True)
args = parser.parse_args()

# ----------------------
# FORCE CPU
# ----------------------
device = "cpu"

# ----------------------
# Load Model
# ----------------------
def load_model_from_config(config, ckpt, device="cpu"):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    # FIX for PyTorch 2.6+
    pl_sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

# ----------------------
# Sampling
# ----------------------
@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1):
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    shape = [4, h // 8, w // 8]

    samples, _ = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=n_samples,
        shape=shape,
        verbose=False,
        start_code=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
    )
    return samples

# ----------------------
# Decode
# ----------------------
def decode_to_im(model, samples, n_samples=1, nrow=1):
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)

    x_sample = 255. * rearrange(
        ims.cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c',
        n1=n_samples // nrow,
        n2=nrow
    )
    return Image.fromarray(x_sample.astype(np.uint8))

# ----------------------
# Read files
# ----------------------
def read_lines_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

image_filenames = read_lines_from_file(args.image_filenames_file)
prompts = read_lines_from_file(args.prompts_file)

# ----------------------
# LIMIT for CPU DEBUG
# ----------------------
nums = min(len(image_filenames), 3)

# ----------------------
# Config
# ----------------------
config = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
ckpt = args.ckpt
feature_npy_path = args.feature_npy_path

# Reduced params for CPU
scale = 3
h = 256
w = 256
ddim_steps = 10
ddim_eta = 0.0

# ----------------------
# Load SD model
# ----------------------
model = load_model_from_config(config, ckpt, device)
sampler = DDIMSampler(model)

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

fusion_model = FusionModel(encode_size=59136, feature_size=63).to(device)

fusion_ckpt = torch.load(ckpt, map_location=device, weights_only=False)
fusion_model.load_state_dict(fusion_ckpt['fusion_model_state_dict'])
fusion_model.eval()

# ----------------------
# Output folder
# ----------------------
os.makedirs(args.img_save_path, exist_ok=True)

# ----------------------
# Generation loop
# ----------------------
for i in tqdm(range(nums), desc="Processing images"):

    prompt = prompts[i]
    torch.manual_seed(0)

    orig_emb = model.get_learned_conditioning([prompt])

    feature = torch.tensor(
        np.load(feature_npy_path),
        dtype=torch.float32
    ).to(device)

    # Fusion
    fused_feature = fusion_model(orig_emb.view(1, -1), feature)
    fused_feature = fused_feature.view(orig_emb.shape)

    # EMA blend
    fused_feature = (1.0 - args.mu) * fused_feature + args.mu * orig_emb

    # Sampling
    start_code = torch.randn((1, 4, h // 8, w // 8))

    samples = sample_model(
        model, sampler, fused_feature,
        h, w, ddim_steps, scale, ddim_eta,
        start_code=start_code,
        n_samples=1
    )

    img = decode_to_im(model, samples)

    img = img.resize((400, 400))

    save_path = os.path.join(args.img_save_path, image_filenames[i])
    img.save(save_path)