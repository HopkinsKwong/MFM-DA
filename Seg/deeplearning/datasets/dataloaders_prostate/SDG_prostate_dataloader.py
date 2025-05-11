from torch.utils import data  # 从torch.utils导入data模块，用于数据加载
import numpy as np  # 导入numpy库，用于数组操作
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
from batchgenerators.utilities.file_and_folder_operations import *  # 从batchgenerators库导入文件和文件夹操作的所有函数
import torch  # 导入PyTorch库

def show(a,id):
    print(f"a.shape:{a.shape}")
    # 将张量移至cpu并转换为numpy数组
    # 如果图像是多通道（比如RGB），先进行维度变换
    # 对于单通道图像可以跳过这一步

    # 反标准化到0-255范围，并转换类型
    a_np = (a * 255).astype(np.uint8)
    # 创建图像对象
    img = Image.fromarray(a_np)
    # 保存图像，指定合适的文件格式（如.png, .jpg等）
    img.save(f'A_label_{id}_image.png')

# 定义一个名为RIGA_labeled_set的类，继承自data.Dataset
class prostate_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(512, 512)):
        super().__init__()  # 调用父类的初始化方法
        self.root = root  # 数据集的根目录
        self.img_list = img_list  # 图像文件列表
        self.label_list = label_list  # 标签文件列表
        self.len = len(img_list)  # 数据集的长度
        self.target_size = target_size  # 图像和标签的目标尺寸

    def __len__(self):
        return self.len  # 返回数据集的长度

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])  # 获取图像文件的路径
        label_file = join(self.root, self.label_list[item])  # 获取标签文件的路径
        img = Image.open(img_file)  # 打开图像文件
        label = Image.open(label_file).convert('L')  # 打开标签文件
        # show(np.array(label), 0)
        # show(np.array(label)>0,1)
        img = img.resize(self.target_size)  # 调整图像尺寸
        label = label.resize(self.target_size, resample=Image.NEAREST)  # 调整标签尺寸，使用最近邻插值
        # show(np.array(label), 2)
        # show(np.array(label) > 0, 3)
        img_npy = np.array(img)  # 将图像转换为numpy数组并调整维度顺序
        # img_npy = np.stack((img_npy,) * 3, axis=-1)
        # print(f"img_npy.shape:{img_npy.shape}")
        img_npy = img_npy.transpose(2, 0, 1).astype(np.float32)
        label_npy = np.array(label)  # 将标签转换为numpy数组
        mask = np.zeros_like(label_npy)  # 创建一个与标签相同形状的全零数组
        mask[label_npy > 0] = 1  # 将标签中大于0的值设为1
        return img_npy, mask[np.newaxis], img_file  # 返回图像、标签和图像文件路径

# 定义一个名为RIGA_unlabeled_set的类，继承自data.Dataset
class RIGA_unlabeled_set(data.Dataset):
    def __init__(self, root, img_list, target_size=(512, 512), img_normalize=True):
        super().__init__()  # 调用父类的初始化方法
        self.root = root  # 数据集的根目录
        self.img_list = img_list  # 图像文件列表
        self.len = len(img_list)  # 数据集的长度
        self.target_size = target_size  # 图像的目标尺寸
        self.img_normalize = img_normalize  # 是否对图像进行归一化

    def __len__(self):
        return self.len  # 返回数据集的长度

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])  # 获取图像文件的路径
        img = Image.open(img_file)  # 打开图像文件
        img = img.resize(self.target_size)  # 调整图像尺寸
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)  # 将图像转换为numpy数组并调整维度顺序
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()  # 对图像进行归一化
        return img_npy, None, img_file  # 返回图像和图像文件路径

# 定义一个名为Cutout的类，用于随机遮挡图像的一部分
class Cutout(object):
    """随机遮挡图像的一部分。

    参数:
        n_holes (int): 每张图像中要遮挡的部分数量。
        length (int): 每个遮挡部分的边长（像素）。
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes  # 遮挡部分的数量
        self.length = length  # 每个遮挡部分的边长

    def __call__(self, img):
        """
        参数:
            img (Tensor): 大小为(C, H, W)的张量图像。
        返回:
            Tensor: 遮挡了n_holes个length x length部分的图像。
        """
        h, w = img.shape[1], img.shape[2]  # 获取图像的高度和宽度

        mask = np.ones((h, w), np.float32)  # 创建一个全1的掩码

        for n in range(self.n_holes):
            y = np.random.randint(h)  # 随机选择一个y坐标
            x = np.random.randint(w)  # 随机选择一个x坐标

            y1 = np.clip(y - self.length // 2, 0, h)  # 计算遮挡部分的上边界
            y2 = np.clip(y + self.length // 2, 0, h)  # 计算遮挡部分的下边界
            x1 = np.clip(x - self.length // 2, 0, w)  # 计算遮挡部分的左边界
            x2 = np.clip(x + self.length // 2, 0, w)  # 计算遮挡部分的右边界

            mask[y1: y2, x1: x2] = 0.  # 将掩码中的对应部分设为0

        mask = np.expand_dims(mask, axis=0)  # 将掩码扩展为(1, H, W)
        mask = np.repeat(mask, img.shape[0], axis=0)  # 在通道维度上复制扩展
        img = img * mask  # 将掩码应用于图像

        return img  # 返回遮挡后的图像

# 定义一个名为Cutout_RIGA_labeled_set的类，继承自data.Dataset
class Cutout_RIGA_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(512, 512), length=64, img_normalize=True):
        super().__init__()  # 调用父类的初始化方法
        self.root = root  # 数据集的根目录
        self.img_list = img_list  # 图像文件列表
        self.label_list = label_list  # 标签文件列表
        self.len = len(img_list)  # 数据集的长度
        self.target_size = target_size  # 图像和标签的目标尺寸
        self.img_normalize = img_normalize  # 是否对图像进行归一化
        self.cutout = Cutout(n_holes=1, length=length)  # 创建一个Cutout实例

    def __len__(self):
        return self.len  # 返回数据集的长度

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])  # 获取图像文件的路径
        label_file = join(self.root, self.label_list[item])  # 获取标签文件的路径
        img = Image.open(img_file)  # 打开图像文件
        label = Image.open(label_file)  # 打开标签文件
        img = img.resize(self.target_size)  # 调整图像尺寸
        label = label.resize(self.target_size, resample=Image.NEAREST)  # 调整标签尺寸，使用最近邻插值
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)  # 将图像转换为numpy数组并调整维度顺序
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()  # 对图像进行归一化

        img_npy = img_npy.transpose(1, 2, 0)  # 将维度转换为(H, W, C)
        img_npy = self.cutout(img_npy)  # 调用Cutout实例的__call__方法
        img_npy = img_npy.transpose(2, 0, 1)  # 将维度转换回(C, H, W)

        label_npy = np.array(label)  # 将标签转换为numpy数组
        mask = np.zeros_like(label_npy)  # 创建一个与标签相同形状的全零数组
        mask[label_npy > 0] = 1  # 将标签中大于0的值设为1
        mask[label_npy == 128] = 2  # 将标签中等于128的值设为2
        return img_npy, mask[np.newaxis], img_file  # 返回图像、标签和图像文件路径