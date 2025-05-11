import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

from deeplearning.datasets.utils.StructureAwareBrightnessAugmentation import Structure_Aware_Brightness_Augmentation
from deeplearning.utils.fourier import FDA_source_to_target_np
from deeplearning.datasets.utils.normalize import normalize_image
from deeplearning.datasets.utils.slaug import LocationScaleAugmentation


def normalize_image_to_0_1_per_sample(img_npy):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            img_npy[b, c] = ((img_npy[b, c] - img_npy[b, c].min()) / (img_npy[b, c].max() - img_npy[b, c].min()))
    return img_npy


def fourier_augmentation_reverse(data, fda_beta=0.1):
    this_fda_beta = round(0.05+np.random.random() * fda_beta, 2)
    lowf_batch = data[::-1]
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data


def sl_augmentation(image, mask):
    location_scale = LocationScaleAugmentation(vrange=(0., 255.), background_threshold=0.01)
    GLA = location_scale.Global_Location_Scale_Augmentation(image.copy())
    LLA = location_scale.Local_Location_Scale_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return GLA, LLA

# def get_train_transform(patch_size=(512, 512)):
#     tr_transforms = []
#     tr_transforms.append(
#         SpatialTransform_2(
#             patch_size, [i // 2 for i in patch_size],
#             do_elastic_deform=True, deformation_scale=(0, 0.25),
#             do_rotation=True,
#             angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
#             angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
#             do_scale=True, scale=(0.75, 1.25),
#             border_mode_data='constant', border_cval_data=0,
#             border_mode_seg='constant', border_cval_seg=0,
#             order_seg=1, order_data=3,
#             random_crop=True,
#             p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
#         )
#     )
#     tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
#     tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
#     # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
#     # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
#     tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
#     # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
#     #                                           p_per_channel=0.5, p_per_sample=0.15))
#     tr_transforms = Compose(tr_transforms)
#     return tr_transforms

def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                              p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_structure_destroyed_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def collate_fn_tr(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    data_dict['data'] = normalize_image(data_dict['data'])
    return data_dict

def collate_fn_tr_SLAug(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    GLA, LLA = sl_augmentation(data_dict['data'], data_dict['seg'])
    data_dict['data'] = normalize_image(data_dict['data'])
    data_dict['GLA'] = normalize_image(GLA)
    data_dict['LLA'] = normalize_image(LLA)
    return data_dict

def collate_fn_ts(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    data_dict['data'] = normalize_image(data_dict['data'])
    return data_dict


def collate_fn_tr_only_sd_trans(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_structure_destroyed_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    data_dict['data'] = normalize_image(data_dict['data'])
    return data_dict

# unet测试用
def collate_fn_tr_only_sd_trans_with_normalize(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_structure_destroyed_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    data_dict['data'] = normalize_image(data_dict['data'])
    return data_dict

def collate_fn_tr_styleaug(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)
    fda_data = fourier_augmentation_reverse(data_dict['data'])
    # data_dict['fda_data'] = normalize_image(fda_data)
    # GLA, _ = sl_augmentation(data_dict['data'], data_dict['seg'])
    # data_dict['data'] = normalize_image(data_dict['data'])
    # data_dict['GLA'] = normalize_image(GLA)

    data_dict['fda_data'] = normalize_image(fda_data)
    GLA, _ = sl_augmentation(data_dict['data'], data_dict['seg'])
    data_dict['data'] = normalize_image(data_dict['data'])
    data_dict['GLA'] = normalize_image(GLA)


    return data_dict

def get_base_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def get_BigAug():
    tr_transforms = []
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def collate_fn_tr_big_fda_gla(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = get_base_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)

    BigAug_transforms = get_BigAug()
    BigAug_data = BigAug_transforms(**data_dict)['data']
    data_dict['BigAug'] = normalize_image(BigAug_data)

    fda_data = fourier_augmentation_reverse(data_dict['data'])
    data_dict['fda_data'] = normalize_image(fda_data)
    GLA, _ = sl_augmentation(data_dict['data'], data_dict['seg'])

    data_dict['data'] = normalize_image(data_dict['data'])
    data_dict['GLA'] = normalize_image(GLA)
    return data_dict

def get_train_transform_SABA(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    # tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15)) ## 使用SABA的时候不使用
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def brightness_augmentation(image, mask):
    BrightnessAug = Structure_Aware_Brightness_Augmentation()
    SABA = BrightnessAug.Local_Brightness_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return SABA

def collate_fn_tr_styleaug_SABAug(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'seg': label, 'name': name}
    tr_transforms = (get_train_transform(patch_size=image.shape[-2:]))
    data_dict = tr_transforms(**data_dict)
    fda_data = fourier_augmentation_reverse(data_dict['data'])
    data_dict['fda_data'] = normalize_image(fda_data)
    GLA, LLA = sl_augmentation(data_dict['data'], data_dict['seg'])
    SABA = brightness_augmentation(data_dict['data'], data_dict['seg'])
    data_dict['data'] = normalize_image(data_dict['data'])
    data_dict['GLA'] = normalize_image(GLA)
    data_dict['LLA'] = normalize_image(LLA)
    data_dict['SABA'] = normalize_image(SABA)
    return data_dict

def STY_collate_fn_tr(batch):
    # 解压batch，获取image1、image2、label和name
    image1, image2, label, name = zip(*batch)
    image1 = np.stack(image1, 0)  # 形状 (B, C, H, W)
    image2 = np.stack(image2, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)

    # 合并image1和image2的通道
    combined_data = np.concatenate([image1, image2], axis=1)  # 新形状 (B, 2C, H, W)

    # 创建数据字典并应用变换
    data_dict = {'data': combined_data, 'seg': label, 'name': name}
    tr_transforms = get_train_transform(patch_size=combined_data.shape[-2:])
    data_dict = tr_transforms(**data_dict)

    # 拆分回image1和image2
    num_channels = image1.shape[1]  # 原image1的通道数
    transformed_combined = data_dict['data']
    image1_transformed = transformed_combined[:, :num_channels, ...]
    image2_transformed = transformed_combined[:, num_channels:, ...]
    # 归一化处理
    data_dict['data'] = normalize_image(image1_transformed)
    data_dict['data_sty'] = normalize_image(image2_transformed)
    return data_dict
