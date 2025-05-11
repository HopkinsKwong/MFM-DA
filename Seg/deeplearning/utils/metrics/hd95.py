import torch
import monai
import medpy.metric.binary as mmb
import numpy as np



def get_HD95_threshold(output, mask, threshold=0.5):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    # print(f"output:{output.shape}")
    # print(f"mask:{mask.shape}")
    # # Check if all elements are 0 or 1
    # unique_elements = torch.unique(output)
    # is_binary = torch.all((unique_elements == 0) | (unique_elements == 1))
    #
    # # Check if the tensor is entirely zero
    # is_all_zeros = torch.all(output == 0)
    #
    # print("Is the tensor binary (only 0s and 1s)?", is_binary.item())
    # print("Is the tensor entirely zeros?", is_all_zeros.item())
    # Check if the entire output tensor is zero

    is_all_zeros = torch.all(output == 0).item()

    if is_all_zeros:
        # Return HD95 as 0 if output is all zeros
        return 0
    else:
        # Convert tensors to NumPy arrays for MONAI compatibility
        output_np = output.cpu().numpy()  # Ensure tensor is on CPU for .numpy()
        mask_np = mask.cpu().numpy()

        # Compute HD95 using MONAI
        HD95 = mmb.hd95(output_np, mask_np)
        # print(f"HD95: {HD95}")

    return HD95


def get_hard_hd95(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        if torch.all(output == 0) or torch.all(mask == 0):
            continue
        dice_list.append(get_HD95_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(dice_list), dice_list
    else:
        return np.mean(dice_list)