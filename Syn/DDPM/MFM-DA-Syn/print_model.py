import torch
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from argparse import ArgumentParser
import os
from torch.optim import Adam
from torchvision.utils import save_image



if __name__ == '__main__':
    image_size=224
    model_target = MyUnet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        self_condition=False,
    ).cuda()

    model_target.prepare(style_condition=True,two_stage_step=300)
    diffusion_target = GaussianDiffusion(
        model_target,
        image_size = image_size,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).to('cuda')

    print(f"diffusion_target:{diffusion_target}")
    print(f"unet:{model_target}")




