import argparse
from deeplearning.utils.file_utils import gen_random_str
from batchgenerators.utilities.file_and_folder_operations import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='Random seed for reproducibility.')
    parser.add_argument('--model', default="Alignment_reins_RETFound_seg_epoch_100_seed42", required=False,
                        help='Model name.')
    parser.add_argument('--chk', default="model_final.model", required=False,
                        help='Checkpoint name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--log_folder', default="/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Seg/deeplearning/log/ours/D/" ,required=False,
                        help='Log folder.')
    parser.add_argument('--tag', default="{}".format(gen_random_str()), required=False,
                        help='Run identifier.')
    parser.add_argument('--inference_tag', default="all", required=False,
                        help='Inference tag.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[224, 224], required=False,
                        help='patch size.')
    parser.add_argument('-r', '--root', default="/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Seg/deeplearning/data/",required=False,
                        help='root folder.')
    parser.add_argument('--ts_csv', nargs='+',
                        default=[

                            "/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Seg/deeplearning/data/DDPM_target_C.csv",
                            "/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Seg/deeplearning/data/DDPM_target_D.csv", ],
                        required=False, help='test csv file.')
    #"/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Seg/deeplearning/data/DDPM_target_A.csv",
                            #"/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Seg/deeplearning/data/DDPM_target_B.csv",

    args = parser.parse_args()
    model_name = args.model
    chk_name = args.chk
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    args.tag = ""
    tag = args.tag
    # log_folder = join(log_folder, model_name+'_'+tag)
    log_folder = join(log_folder, model_name)
    patch_size = tuple(args.patch_size)
    root_folder = args.root
    ts_csv = tuple(args.ts_csv)
    inference_tag = args.inference_tag
    # now_seed = 42


    if model_name == 'unet_coral_epoch_100_seed42':
        from deeplearning.inference.inference_nets.inference_unet import inference
    elif model_name in 'SETR_PUP':
        from deeplearning.inference.inference_nets.inference_SETR_PUP import inference
    elif model_name in 'H2Former':
        from deeplearning.inference.inference_nets.inference_H2Former import inference
    elif model_name == 'Alignment_reins_RETFound_seg_epoch_100_seed42':
        from deeplearning.inference.inference_nets.inference_reins_RETFound_seg import inference
    elif model_name == 'RETFound_seg_Cora_epoch_100_seed42':
        from deeplearning.inference.inference_nets.inference_RETFound_seg import inference
    else:
        print('No model named "{}"!'.format(model_name))
        return

    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference(chk_name, gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag)


if __name__ == '__main__':
    main()
