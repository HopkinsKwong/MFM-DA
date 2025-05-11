import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def get_hard_hd95(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)

    hd95_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]

        hd95_value = calculate_hd95(output.cpu().numpy(), mask.cpu().numpy())
        if hd95_value == float('inf'):
            hd95_value = 0
        #mask像素和过小时异常值的处理，根据数据集情况自己设定，代表非常差的分割效果
        hd95_list.append(hd95_value)

    if return_list:
        return np.mean(hd95_list), hd95_list
    else:
        return np.mean(hd95_list)


def calculate_hd95(pred, gt):
    """
    Calculate the Hausdorff Distance 95% between the predicted binary mask and the ground truth mask.
    Both inputs must be binary masks.
    """
    # Ensure binary masks
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    # Get the coordinates of the boundary points of the prediction and ground truth
    pred_points = np.argwhere(pred == 1)
    gt_points = np.argwhere(gt == 1)

    if len(pred_points) == 0 or len(gt_points) == 0:
        # If either prediction or ground truth is empty, return a large number as HD95
        return np.inf

    # Compute directed Hausdorff distances
    forward_hausdorff = directed_hausdorff(pred_points, gt_points)[0]
    backward_hausdorff = directed_hausdorff(gt_points, pred_points)[0]

    # Combine both forward and backward distances
    hausdorff_distances = [forward_hausdorff, backward_hausdorff]

    # Calculate 95% of Hausdorff distance
    hd95_value = np.percentile(hausdorff_distances, 95)

    return hd95_value
