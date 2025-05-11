import torch
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import os
from torch.optim import Adam
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--output_path', type=str,default=None, help='Path to output directory')
parser.add_argument('--ckpt_path', type=str,default=None, help='Path to ckpt directory')

opts = parser.parse_args()
image_size=224
model = MyUnet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)
model.prepare()
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)


ckpt=torch.load(opts.ckpt_path,map_location='cpu')
for key in ckpt.keys():
    print(key)
diffusion.load_state_dict(ckpt)
print('success')

sampled_images = diffusion.sample(batch_size = 16)
# 指定保存图像的目录
output_dir = opts.output_path
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

# 循环遍历Tensor，保存每张图像
for i in range(output.shape[0]):
    # 构建每张图像的保存路径
    save_path = os.path.join(output_dir, f'image_{i + 1}.png')
    # 保存图像
    save_image(output[i], save_path)

print(f'Images saved in {output_dir}')