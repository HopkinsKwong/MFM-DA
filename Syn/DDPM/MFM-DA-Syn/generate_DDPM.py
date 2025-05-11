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
parser = ArgumentParser()
# parser.add_argument('--img_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--save_path', type=str,default=None, help='Path to input directory')
parser.add_argument('--ckpt_path', type=str,default=None, help='Path to ckpt')
parser.add_argument('--image_num', type=int,default=None, help='Number of generated image')
opts = parser.parse_args()
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
ckpt=torch.load(opts.ckpt_path,map_location='cpu', weights_only=True)
diffusion_target.load_state_dict(ckpt,strict=False)

from tqdm import tqdm

output_dir=opts.save_path
image_num = opts.image_num
# 假设 diffusion_target 和 output_dir 已经定义
# 定义批量大小
batch_size = 16  # 根据你的 GPU 内存调整 batch_size

# 计算需要的批次数
batch_num = (image_num + batch_size - 1) // batch_size  # 向上取整

# 使用 tqdm 显示进度条
for batch_idx in tqdm(range(batch_num), desc="Generating Images", unit="batch"):
    # 生成一批图像
    sampled_images_batch = diffusion_target.sample(batch_size=batch_size)

    # 获取当前批次中需要处理的图像数量（最后一组可能不足 batch_size）
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, image_num)
    current_batch_size = end_idx - start_idx

    # 将批量图像拆分为单张图像并保存
    for i in range(current_batch_size):
        global_idx = start_idx + i
        single_image = sampled_images_batch[i]  # 提取单张图像

        # 定义保存路径
        save_path = os.path.join(output_dir, 'images', f'image_{global_idx}.png')

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存单张图像
        save_image(single_image, save_path)