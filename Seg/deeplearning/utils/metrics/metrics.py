# The HD95 calculation code in the following section is borrowed from the metric.py file of the medpy library.
# medpy project repository: https://github.com/loli/medpy

import numpy as np
import torch


def get_dice_threshold(output, mask, threshold=0.01):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice

def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value

def get_hard_dice(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_list.append(get_dice_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(dice_list), dice_list
    else:
        return np.mean(dice_list)

# IOU实现
def get_iou_threshold(output, mask, threshold=0.5):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: iou of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (mask * output).sum()
    union = (mask + output).sum() - intersection
    iou = intersection / union if union > 0 else 0.0

    return iou

def get_hard_iou(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_list.append(get_iou_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(iou_list), iou_list
    else:
        return np.mean(iou_list)

