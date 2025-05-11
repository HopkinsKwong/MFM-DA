from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
import torch


class CHAOST2_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        label_file = join(self.root, self.label_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file)
        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        label_npy = np.array(label)
        mask = np.zeros_like(label_npy)
        mask[label_npy > 0] = 1
        mask[label_npy == 128] = 2
        return img_npy, mask[np.newaxis], img_file


class CHAOST2_unlabeled_set(data.Dataset):
    def __init__(self, root, img_list, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        img = Image.open(img_file)
        img = img.resize(self.target_size)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        return img_npy, None, img_file


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h, w = img.shape[1], img.shape[2]  # 获取图像的高度和宽度

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        # mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        mask = np.expand_dims(mask, axis=0)  # 将 mask 扩展为 (1, H, W)
        mask = np.repeat(mask, img.shape[0], axis=0)  # 在通道维度上复制扩展
        img = img * mask

        return img


class Cutout_CHAOST2_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(512, 512), length=64,img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize
        self.cutout=Cutout(n_holes=1,length=length)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])
        label_file = join(self.root, self.label_list[item])
        img = Image.open(img_file)
        label = Image.open(label_file)
        img = img.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()

        # 将维度转换为 (H, W, C)
        img_npy = img_npy.transpose(1, 2, 0)
        img_npy = self.cutout(img_npy)  # 调用 Cutout 实例的 __call__ 方法
        # 将维度转换回 (C, H, W)
        img_npy = img_npy.transpose(2, 0, 1)

        label_npy = np.array(label)
        mask = np.zeros_like(label_npy)
        mask[label_npy > 0] = 1
        mask[label_npy == 128] = 2
        return img_npy, mask[np.newaxis], img_file