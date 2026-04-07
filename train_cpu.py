from omegaconf import OmegaConf
import torch
from torch import nn
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
# Arguments
# ----------------------
parser = argparse.ArgumentParser(description='Train the model (CPU version)')
parser.add_argument('--gesture', type=str, required=True, help='Gesture name')
parser.add_argument('--Lambda', type=float, default=0.7, help='Hyperparameter Lambda')
args = parser.parse_args()

# ----------------------
# Device (FORCE CPU)
# ----------------------
device = "cpu"

# ----------------------
# Model Loader
# ----------------------
def load_model_from_config(config, ckpt, device="cpu"):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

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

    samples_ddim, _ = sampler.sample(
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
    return samples_ddim

# ----------------------
# Image utils
# ----------------------
def load_img(path, target_size=512):
    image = Image.open(path).convert("RGB")
    tform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    image = tform(image)
    return 2. * image - 1.

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
# Utils
# ----------------------
def read_lines_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

# ----------------------
# Paths
# ----------------------
image_folder = f"./{args.gesture}"
image_filenames_file = f"./{args.gesture}_BLIP2_file_name.txt"
prompts_file = f"./{args.gesture}_BLIP2_modified.txt"
feature_npy_path = f'./{args.gesture}.npy'

image_filenames = read_lines_from_file(image_filenames_file)
prompts = read_lines_from_file(prompts_file)

# ----------------------
# DEBUG MODE (reduce workload)
# ----------------------
nums = min(len(image_filenames), 3)

# ----------------------
# Stable Diffusion setup
# ----------------------
config = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
ckpt = "./stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"

model = load_model_from_config(config, ckpt, device)
sampler = DDIMSampler(model)

fusion_model = FusionModel(encode_size=59136, feature_size=63).to(device)

# ----------------------
# Hyperparameters (reduced for CPU)
# ----------------------
scale = 3
h = 192
w = 192
ddim_steps = 5   # reduced
ddim_eta = 0.0

# ----------------------
# Training Loop
# ----------------------
for idx in tqdm(range(nums), desc="Processing images"):

    image_path = os.path.join(image_folder, image_filenames[idx])
    image = load_img(image_path).unsqueeze(0).to(device)
    prompt = prompts[idx]

    torch.manual_seed(0)

    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(image)
    ).detach()

    orig_embs = model.get_learned_conditioning([prompt]).detach()

    feature = torch.tensor(
        np.load(feature_npy_path),
        dtype=torch.float32
    ).to(device)

    # ----------------------
    # Fusion
    # ----------------------
    fused_feature_orig = fusion_model(orig_embs.view(1, -1), feature)
    fused_feature_orig = fused_feature_orig.view(orig_embs.shape)

    fused_feature = (
        args.Lambda * orig_embs +
        (1.0 - args.Lambda) * fused_feature_orig
    ).clone().detach().requires_grad_(True)

    # ----------------------
    # Text Embedding Optimization
    # ----------------------
    opt = torch.optim.Adam([fused_feature], lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(1):  # reduced iterations
        opt.zero_grad()

        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(1000, (1,))
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, fused_feature)
        loss = loss_fn(pred_noise, noise)

        loss.backward()
        opt.step()

    fused_feature.requires_grad = False
    model.train()

    # ----------------------
    # Model Fine-tuning
    # ----------------------
    opt = torch.optim.Adam(model.model.parameters(), lr=1e-6)

    for _ in range(1):  # reduced iterations
        opt.zero_grad()

        noise = torch.randn_like(init_latent)
        t_enc = torch.randint(model.num_timesteps, (1,))
        z = model.q_sample(init_latent, t_enc, noise=noise)

        pred_noise = model.apply_model(z, t_enc, fused_feature)
        loss = loss_fn(pred_noise, noise)

        loss.backward()
        opt.step()

# ----------------------
# Save checkpoint
# ----------------------
torch.save({
    'state_dict': model.state_dict(),
    'fusion_model_state_dict': fusion_model.state_dict()
}, "./model_finetuned_cpu.ckpt")