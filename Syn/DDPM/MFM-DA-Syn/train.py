import torch
from unet import MyUnet
from model.ddpm import GaussianDiffusion
from torch.nn import DataParallel
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torch.optim import Adam
from argparse import ArgumentParser
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

parser = ArgumentParser()
parser.add_argument('--data_path', type=str, default=None, help='Path to input directory')

opts = parser.parse_args()
image_size = 224
model = MyUnet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)
model.prepare()
diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
)


class Train_Data(Dataset):
    def __init__(self, img_path):
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([image_size, image_size])
        ])
        self.data_path = img_path
        self.file_names = os.listdir(self.data_path)
        self.l = len(self.file_names)

    def __getitem__(self, idx):
        idx = idx % len(self.file_names)
        image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
        image = self.loader(image)
        return image

    def __len__(self):
        return self.l


batch_size = 64
real_data = Train_Data(opts.data_path)
real_dataloader = DataLoader(real_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=30,
                             drop_last=True)

diffusion = DataParallel(diffusion).to('cuda')
print('success')
optizer = Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
global_step = 0
dir = 'output/'
os.makedirs(dir + 'models', exist_ok=True)

total_epochs = 700000
for epoch in range(total_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(real_dataloader):
        image = batch.to('cuda')
        loss = diffusion(image)
        optizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optizer.step()
        total_loss += loss.item()
        global_step += 1
    average_loss = total_loss / len(real_dataloader)
    print(f"epoch:{epoch} - average loss:{average_loss}")
    # 每50个epoch保存一次模型和采样图像
    if (epoch + 1) % 1000 == 0:
        epoch_dir = os.path.join(dir, f'epoch_{epoch + 1}')
        os.makedirs(epoch_dir, exist_ok=True)

        # 保存模型权重
        model_save_path = os.path.join(epoch_dir, f'model_{epoch + 1}.pth')
        torch.save(diffusion.module.state_dict(), model_save_path)

        # 生成和保存图像
        sampled_images = diffusion.module.sample(batch_size=16)
        for i, img in enumerate(sampled_images):
            save_path = os.path.join(epoch_dir, f'image_{epoch + 1}_{i + 1}.png')
            save_image(img, save_path)

        print(f'Model and images saved for epoch {epoch + 1} in {epoch_dir}')

print(f'Training completed. All models and images are saved in {dir}')
