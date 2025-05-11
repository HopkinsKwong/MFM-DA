from time import time
from fvcore.nn import FlopCountAnalysis
from torch.cuda.amp import autocast

from deeplearning.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from deeplearning.models.unet import UNet
from deeplearning.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, RIGA_unlabeled_set
from deeplearning.datasets.utils.convert_csv_to_list import convert_labeled_list
from deeplearning.datasets.utils.transform import collate_fn_ts
from deeplearning.utils.metrics.metrics import *
from deeplearning.utils.visualization import visualization_as_nii
from deeplearning.utils.metrics.hd95 import get_hard_hd95

import numpy as np
from PIL import Image
import numpy as np

def blend_image_and_label(image_np, label_pixel):

    # 转换标签数据形状为 (H, W)
    if label_pixel.ndim == 3:  # 如果是 (C, H, W) 或 (1, H, W)
        label_pixel = label_pixel.squeeze(0)  # 压缩通道维度

    image = Image.open(image_np)
    image = image.resize((224,224))

    label = Image.fromarray(label_pixel)

    width, height = label.size
    rgba_image = Image.new("RGBA", (width, height))

    for y in range(height):
        for x in range(width):
            pixel_value = label.getpixel((x, y))
            if pixel_value == 128:
                rgba_image.putpixel((x, y), (90, 138, 144, 175))  # 视杯（蓝色）
            elif pixel_value == 255:
                rgba_image.putpixel((x, y), (150, 191, 47, 175))  # 视盘（绿色）
            else:
                rgba_image.putpixel((x, y), (0, 0, 0, 0))         # 透明背景

    image = image.convert("RGBA")
    # print(f"image.shape:{np.array(image).shape}")
    # print(f"rgba_image.shape:{np.array(rgba_image).shape}")
    result_image = Image.alpha_composite(image, rgba_image)
    return result_image


def inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, inference_tag='all'):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    visualization_folder = join(visualization_folder, inference_tag)
    maybe_mkdir_p(visualization_folder)

    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    if ts_label_list is None:
        evaluate = False
        ts_dataset = RIGA_unlabeled_set(root_folder, ts_img_list, patch_size)
    else:
        evaluate = True
        ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)
    ts_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                batch_size=4,
                                                num_workers=2,
                                                shuffle=False,
                                                pin_memory=True,
                                                collate_fn=collate_fn_ts)
    model = UNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    assert isfile(join(model_folder, chk_name)), 'missing model checkpoint {}!'.format(join(model_folder, chk_name))
    params = torch.load(join(model_folder, chk_name), weights_only=True)
    model.load_state_dict(params['model_state_dict'])

    # Calculate model total parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total Parameters: {num_params}")

    # Calculate model FLOPs
    input_tensor = torch.randn(4, 3, 224, 224).to(device)
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total()}")

    time_per_image_cost_list = []
    memory_usage_list = []

    seg_list = list()
    output_list = list()
    data_list = list()
    name_list = list()
    with torch.no_grad():
        model.eval()
        for iter, batch in enumerate(ts_dataloader):
            batch_start_time = time()

            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            name = batch['name']
            with autocast():
                output = model(data)
            output_sigmoid = torch.sigmoid(output).cpu().numpy()
            seg_list.append(batch['seg'])
            output_list.append(output_sigmoid)
            data_list.append(batch['data'])
            name_list.append(name)

            memory_usage_list.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

            batch_end_time = time()
            batch_time = batch_end_time - batch_start_time
            time_per_image = batch_time / data.size(0)
            time_per_image_cost_list.append(time_per_image)

    print(f'Average Time per image: {np.mean(time_per_image_cost_list):.4f} seconds')
    print(f"Average Max memory allocated: {np.mean(memory_usage_list):.2f} MB")

    all_data = list()
    all_seg = list()
    all_output = list()
    all_name = list()
    for i in range(len(data_list)):
        for j in range(data_list[i].shape[0]):
            all_data.append(data_list[i][j])
            all_seg.append(seg_list[i][j])
            all_output.append(output_list[i][j])
            all_name.append(name_list[i][j])

            if i in [0,1,2,3,4,5]:
                # 获取当前样本的数据、真实标签和预测输出
                seg_j = seg_list[i][j]  # 形状 (H, W)
                output_j = output_list[i][j]  # 形状 (n, H, W)

                # 处理真实标签（seg_j的值为0,1,2 → 转换为0,128,255）
                seg_pixel = np.zeros_like(seg_j, dtype=np.uint8)
                seg_pixel[seg_j == 1] = 255  # 视盘
                seg_pixel[seg_j == 2] = 128  # 视杯

                # 处理预测标签
                pred_pixel = np.zeros_like(output_j[0], dtype=np.uint8)
                pred_pixel[output_j[0] > 0.5] = 255  # 预测视盘
                pred_pixel[output_j[1] > 0.5] = 128  # 预测视杯
                # 获取文件名并保存
                current_name = name_list[i][j]
                # 生成可视化结果
                true_vis = blend_image_and_label(current_name, seg_pixel)
                pred_vis = blend_image_and_label(current_name, pred_pixel)


                base_name = os.path.basename(current_name)
                file_name_without_ext = os.path.splitext(base_name)[0]

                # 保存真实标签可视化
                true_vis_path = os.path.join(visualization_folder, f"{file_name_without_ext}_true.png")
                true_vis.save(true_vis_path)

                # 保存预测结果可视化
                pred_vis_path = os.path.join(visualization_folder, f"{file_name_without_ext}_pred.png")
                pred_vis.save(pred_vis_path)
    all_data = np.stack(all_data)
    all_seg = np.stack(all_seg)
    all_output = np.stack(all_output)
    visualization_as_nii(all_data[:, 0].astype(np.float32), join(visualization_folder, 'data_channel0.nii.gz'))
    visualization_as_nii(all_data[:, 1].astype(np.float32), join(visualization_folder, 'data_channel1.nii.gz'))
    visualization_as_nii(all_data[:, 2].astype(np.float32), join(visualization_folder, 'data_channel2.nii.gz'))
    visualization_as_nii(all_output[:, 0].astype(np.float32), join(visualization_folder, 'output_disc.nii.gz'))
    visualization_as_nii(all_output[:, 1].astype(np.float32), join(visualization_folder, 'output_cup.nii.gz'))
    if evaluate:
        visualization_as_nii(all_seg[:, 0].astype(np.float32), join(visualization_folder, 'seg.nii.gz'))
        disc_dice, disc_dice_list = get_hard_dice(torch.from_numpy(all_output[:, 0]),
                                                  torch.from_numpy(((all_seg[:, 0] > 0) * 1.0)), return_list=True)
        cup_dice, cup_dice_list = get_hard_dice(torch.from_numpy(all_output[:, 1]),
                                                torch.from_numpy(((all_seg[:, 0] > 1) * 1.0)), return_list=True)

        disc_iou, disc_iou_list = get_hard_iou(torch.from_numpy(all_output[:, 0]),
                                               torch.from_numpy(((all_seg[:, 0] > 0) * 1.0)), return_list=True)
        cup_iou, cup_iou_list = get_hard_iou(torch.from_numpy(all_output[:, 1]),
                                             torch.from_numpy(((all_seg[:, 0] > 1) * 1.0)), return_list=True)
        disc_hd95, disc_hd95_list = get_hard_hd95(torch.from_numpy(all_output[:, 0]),
                                                  torch.from_numpy(((all_seg[:, 0] > 0) * 1.0)), return_list=True)
        cup_hd95, cup_hd95_list = get_hard_hd95(torch.from_numpy(all_output[:, 1]),
                                                torch.from_numpy(((all_seg[:, 0] > 1) * 1.0)), return_list=True)

        # 计算标准差
        disc_dice_std = np.std(disc_dice_list)
        cup_dice_std = np.std(cup_dice_list)
        disc_iou_std = np.std(disc_iou_list)
        cup_iou_std = np.std(cup_iou_list)
        disc_hd95_std = np.std(disc_hd95_list)
        cup_hd95_std = np.std(cup_hd95_list)

        # 格式化输出
        metrics_str = ('Tag: {}\n'
                       '  Disc dice: {} (std: {}); Cup dice: {} (std: {});\n'
                       '  Disc IOU: {} (std: {}); Cup IOU: {} (std: {});\n'
                       '  Disc HD95: {} (std: {}); Cup HD95: {} (std: {})\n').format(
            inference_tag,
            disc_dice, disc_dice_std, cup_dice, cup_dice_std,
            disc_iou, disc_iou_std, cup_iou, cup_iou_std,
            disc_hd95, disc_hd95_std, cup_hd95, cup_hd95_std)


        print(metrics_str)
        with open(join(metrics_folder, '{}.txt'.format(inference_tag)), 'w') as f:
            f.write(metrics_str)
        with open(join(metrics_folder, '{}.csv'.format(inference_tag)), 'w') as f:
            for dice_i in range(len(disc_dice_list)):
                f.write(f'{all_name[dice_i]},disc_dice,{disc_dice_list[dice_i]}\n')
                f.write(f'{all_name[dice_i]},cup_dice,{cup_dice_list[dice_i]}\n')
                f.write(f'{all_name[dice_i]},disc_iou,{disc_iou_list[dice_i]}\n')
                f.write(f'{all_name[dice_i]},cup_iou,{cup_iou_list[dice_i]}\n')
                f.write(f'{all_name[dice_i]},disc_hd95,{disc_hd95_list[dice_i]}\n')
                f.write(f'{all_name[dice_i]},cup_hd95,{cup_hd95_list[dice_i]}\n')
