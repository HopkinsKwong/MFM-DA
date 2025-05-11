from torch.utils import data  # 从torch.utils导入data模块，用于数据加载
import numpy as np  # 导入numpy库，用于数组操作
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
from batchgenerators.utilities.file_and_folder_operations import *  # 从batchgenerators库导入文件和文件夹操作的所有函数
import torch  # 导入PyTorch库
import numpy as np


# 定义一个名为RIGA_labeled_set的类，继承自data.Dataset
class RIGA_labeled_set(data.Dataset):
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
        img = img.resize(self.target_size)  # 调整图像尺寸
        label = label.resize(self.target_size, resample=Image.NEAREST)  # 调整标签尺寸，使用最近邻插值
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)  # 将图像转换为numpy数组并调整维度顺序
        label_npy = np.array(label)  # 将标签转换为numpy数组
        mask = np.zeros_like(label_npy)  # 创建一个与标签相同形状的全零数组
        mask[label_npy > 0] = 1  # 将标签中大于0的值设为1
        mask[label_npy == 128] = 2  # 将标签中等于128的值设为2
        # mask[label_npy < 255] = 1 # 视盘
        # mask[label_npy == 0] = 2  # 视杯
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


class STY_RIGA_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, STY_list, target_size=(512, 512)):
        super().__init__()  # 调用父类的初始化方法
        self.root = root  # 数据集的根目录
        self.img_list = img_list  # 图像文件列表
        self.label_list = label_list  # 标签文件列表
        self.STY_list = STY_list
        self.len = len(img_list)  # 数据集的长度
        self.target_size = target_size  # 图像和标签的目标尺寸

    def __len__(self):
        return self.len  # 返回数据集的长度

    def __getitem__(self, item):
        img_file = join(self.root, self.img_list[item])  # 获取图像文件的路径
        label_file = join(self.root, self.label_list[item])  # 获取标签文件的路径
        STY_file = join(self.root, self.STY_list[item])
        img = Image.open(img_file)  # 打开图像文件
        STY = Image.open(STY_file)
        label = Image.open(label_file).convert('L')  # 打开标签文件
        img = img.resize(self.target_size)  # 调整图像尺寸
        STY = STY.resize(self.target_size)
        label = label.resize(self.target_size, resample=Image.NEAREST)  # 调整标签尺寸，使用最近邻插值
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)  # 将图像转换为numpy数组并调整维度顺序
        STY_npy = np.array(STY).transpose(2, 0, 1).astype(np.float32)
        label_npy = np.array(label)  # 将标签转换为numpy数组
        mask = np.zeros_like(label_npy)  # 创建一个与标签相同形状的全零数组
        mask[label_npy > 0] = 1  # 将标签中大于0的值设为1
        mask[label_npy == 128] = 2  # 将标签中等于128的值设为2
        # mask[label_npy < 255] = 1 # 视盘
        # mask[label_npy == 0] = 2  # 视杯
        return img_npy, STY_npy, mask[np.newaxis], img_file  # 返回图像、标签和图像文件路径


# 定义一个名为RIGA_labeled_set的类，继承自data.Dataset
class FDA_RIGA_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, tag="B",target_size=(512, 512)):
        super().__init__()  # 调用父类的初始化方法
        self.root = root  # 数据集的根目录
        self.img_list = img_list  # 图像文件列表
        self.label_list = label_list  # 标签文件列表
        self.len = len(img_list)  # 数据集的长度
        self.target_size = target_size  # 图像和标签的目标尺寸
        self.tag = tag
        self.target_list = self.get_target_img_by_tag(self.tag)

    def __len__(self):
        return self.len  # 返回数据集的长度

    def get_target_img_by_tag(self,dataset_tag):
        if dataset_tag == "B":
            dataset_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Syn/DDPM/REFUGE_test_10/"
        elif dataset_tag =="C":
            dataset_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Syn/DDPM/BinRushed_10/"
        elif dataset_tag == "D":
            dataset_path = "/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Syn/DDPM/MESSIDIOR_Base2_10/"
            # 获取文件名列表
        file_list = os.listdir(dataset_path)
        # 拼接完整路径
        full_paths = [os.path.join(dataset_path, file_name) for file_name in file_list]
        return full_paths

    def __getitem__(self, item):
        target_file = self.target_list[item%10]
        target = Image.open(target_file)
        target = target.resize(self.target_size)
        target_npy = np.array(target).transpose(2, 0, 1).astype(np.float32)

        img_file = join(self.root, self.img_list[item])  # 获取图像文件的路径
        label_file = join(self.root, self.label_list[item])  # 获取标签文件的路径
        img = Image.open(img_file)  # 打开图像文件
        label = Image.open(label_file).convert('L')  # 打开标签文件
        img = img.resize(self.target_size)  # 调整图像尺寸
        label = label.resize(self.target_size, resample=Image.NEAREST)  # 调整标签尺寸，使用最近邻插值
        img_npy = np.array(img).transpose(2, 0, 1).astype(np.float32)  # 将图像转换为numpy数组并调整维度顺序

        img_npy = FDA_source_to_target_np(img_npy, target_npy, L=0.01)

        label_npy = np.array(label)  # 将标签转换为numpy数组
        mask = np.zeros_like(label_npy)  # 创建一个与标签相同形状的全零数组
        mask[label_npy > 0] = 1  # 将标签中大于0的值设为1
        mask[label_npy == 128] = 2  # 将标签中等于128的值设为2
        # mask[label_npy < 255] = 1 # 视盘
        # mask[label_npy == 0] = 2  # 视杯
        return img_npy, mask[np.newaxis], img_file  # 返回图像、标签和图像文件路径




def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False )
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg