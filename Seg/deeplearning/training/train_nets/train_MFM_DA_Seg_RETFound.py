from time import time
from fvcore.nn import FlopCountAnalysis
from torch.amp import autocast

# Other imports
import torch.nn as nn

from deeplearning.utils.metrics.scipy_hd95 import get_hard_hd95
from deeplearning.utils.file_utils import check_folders
from batchgenerators.utilities.file_and_folder_operations import *
from deeplearning.models.reins_RETFound_seg import Reins_SegHead
from deeplearning.datasets.dataloaders.RIGA_dataloader import RIGA_labeled_set, STY_RIGA_labeled_set
from deeplearning.datasets.utils.convert_csv_to_list import convert_labeled_list,STY_convert_labeled_list
from deeplearning.datasets.utils.transform import STY_collate_fn_tr, collate_fn_ts
from deeplearning.utils.lr import adjust_learning_rate
from deeplearning.utils.metrics.metrics import *
from torchvision.utils import make_grid


def channel_spatial_loss(tensor1, tensor2):
    B,C = tensor1.shape[0],tensor1.shape[1]
    tensor1_flat = tensor1.view(B, C, -1)
    tensor2_flat = tensor2.view(B, C, -1)

    cos_sim = nn.CosineSimilarity(dim=1)
    similarity = cos_sim(tensor1_flat, tensor2_flat)  # [4, 16384]
    loss = 1 - similarity.mean()
    return loss

def train(args):
    model_name = args.model
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    batch_size = args.batch_size
    initial_lr = args.initial_lr
    save_interval = args.save_interval
    num_epochs = args.num_epochs
    continue_training = args.continue_training
    num_threads = args.num_threads
    root_folder = args.root
    tr_csv = tuple(args.tr_csv)
    ts_csv = tuple(args.ts_csv)
    shuffle = not args.no_shuffle

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu])
    tensorboard_folder, model_folder, visualization_folder, metrics_folder = check_folders(log_folder)
    # writer = SummaryWriter(log_dir=tensorboard_folder)

    # 构建数据集
    tr_img_list, tr_label_list, tr_sty_list = STY_convert_labeled_list(tr_csv, r=1)
    tr_dataset = STY_RIGA_labeled_set(root_folder, tr_img_list, tr_label_list, tr_sty_list, patch_size)
    ts_img_list, ts_label_list = convert_labeled_list(ts_csv, r=1)
    ts_dataset = RIGA_labeled_set(root_folder, ts_img_list, ts_label_list, patch_size)

    # 构建dataloader
    train_dataloader = torch.utils.data.DataLoader(tr_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_threads,
                                                   shuffle=shuffle,
                                                   pin_memory=True,
                                                   collate_fn=STY_collate_fn_tr)

    test_dataloader = torch.utils.data.DataLoader(ts_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=num_threads // 2,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn_ts)

    model = Reins_SegHead()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.99, nesterov=True)
    optimizer_encoder = torch.optim.SGD(list(model.rein_RETFound.parameters()), lr=initial_lr, momentum=0.99, nesterov=True)


    start_epoch = 0
    if continue_training:
        assert isfile(join(model_folder, 'model_latest.model')), 'missing model checkpoint!'
        params = torch.load(join(model_folder, 'model_latest.model'))
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        start_epoch = params['epoch']
    print('start epoch: {}'.format(start_epoch))

    amp_grad_scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    # Calculate model total parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total Parameters: {num_params}")

    # # Calculate model FLOPs
    # input_tensor = torch.randn(8, 3, 224, 224).to(device)
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"FLOPs: {flops.total()}")

    start = time()
    time_per_image_cost_list = []
    memory_usage_list = []

    for epoch in range(start_epoch, num_epochs):
        print(args.model + ' Epoch {}:'.format(epoch))
        start_epoch = time()
        model.train()
        lr = adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)
        print('  lr: {}'.format(lr))

        train_loss_list = list()
        train_disc_dice_list = list()
        train_cup_dice_list = list()
        train_disc_IOU_list = list()
        train_cup_IOU_list = list()
        # train_disc_HD95_list = list()
        # train_cup_HD95_list = list()

        for iter, batch in enumerate(train_dataloader):
            batch_start_time = time()

            data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
            data_sty = torch.from_numpy(batch['data_sty']).cuda().to(dtype=torch.float32)
            seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)

            optimizer_encoder.zero_grad()
            with autocast("cuda"):
                f_data = model.rein_RETFound(data)
                f_data_sty = model.rein_RETFound(data_sty)
                alignment_loss = (channel_spatial_loss(f_data[0],f_data_sty[0])
                                  + channel_spatial_loss(f_data[1],f_data_sty[1])
                                  + channel_spatial_loss(f_data[2],f_data_sty[2])
                                  + channel_spatial_loss(f_data[3],f_data_sty[3])) / 4
            amp_grad_scaler.scale(alignment_loss).backward()
            amp_grad_scaler.unscale_(optimizer_encoder)
            amp_grad_scaler.step(optimizer_encoder)
            amp_grad_scaler.update()

            optimizer.zero_grad()
            with autocast("cuda"):
                output = model(data)
                loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            memory_usage_list.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

            optimizer.zero_grad()
            with autocast("cuda"):
                output = model(data_sty)
                loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1], (seg[:, 0] == 2) * 1.0)
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            memory_usage_list.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

            batch_end_time = time()
            batch_time = batch_end_time - batch_start_time
            time_per_image = batch_time / batch_size
            time_per_image_cost_list.append(time_per_image)

            train_loss_list.append(loss.detach().cpu().numpy())
            output_sigmoid = torch.sigmoid(output)
            train_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            train_disc_IOU_list.append(get_hard_iou(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            train_cup_IOU_list.append(get_hard_iou(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            # train_disc_HD95_list.append(
            #     get_hard_hd95(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
            # train_cup_HD95_list.append(
            #     get_hard_hd95(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
            del seg

        mean_tr_loss = np.mean(train_loss_list)
        mean_tr_disc_dice = np.mean(train_disc_dice_list)
        mean_tr_cup_dice = np.mean(train_cup_dice_list)
        mean_tr_disc_IOU = np.mean(train_disc_IOU_list)
        mean_tr_cup_IOU = np.mean(train_cup_IOU_list)
        # mean_tr_disc_HD95 = np.mean(train_disc_HD95_list)
        # mean_tr_cup_HD95 = np.mean(train_cup_HD95_list)

        # writer.add_scalar("Train Scalars/Learning Rate", lr, epoch)
        # writer.add_scalar("Train Scalars/Train Loss", mean_tr_loss, epoch)
        # writer.add_scalar("Train Scalars/Disc Dice", mean_tr_disc_dice, epoch)
        # writer.add_scalar("Train Scalars/Cup Dice", mean_tr_cup_dice, epoch)
        # writer.add_scalar("Train Scalars/Cup IOU", mean_tr_cup_IOU, epoch)
        # writer.add_scalar("Train Scalars/Disc IOU", mean_tr_disc_IOU, epoch)
        # writer.add_scalar("Train Scalars/Cup HD95", mean_tr_cup_HD95, epoch)
        # writer.add_scalar("Train Scalars/Disc HD95", mean_tr_disc_HD95, epoch)

        print('  Tr loss: {}\n'
              '  Tr disc dice: {}; Cup dice: {}\n'
              '  Tr disc IOU: {}; Cup IOU: {}\n'.format(mean_tr_loss, mean_tr_disc_dice, mean_tr_cup_dice,
                                                        mean_tr_disc_IOU, mean_tr_cup_IOU))

        if epoch % 30 == 0:

            val_loss_list, val_disc_dice_list, val_cup_dice_list, val_disc_IOU_list, val_cup_IOU_list = [], [], [], [], []
            val_time_per_image_cost_list = []
            val_memory_usage_list = []

            with torch.no_grad():
                model.eval()
                for iter, batch in enumerate(test_dataloader):
                    batch_start_time = time()

                    data = torch.from_numpy(batch['data']).cuda().to(dtype=torch.float32)
                    seg = torch.from_numpy(batch['seg']).cuda().to(dtype=torch.float32)
                    with autocast("cuda"):
                        output = model(data)
                        loss = criterion(output[:, 0], (seg[:, 0] > 0) * 1.0) + criterion(output[:, 1],
                                                                                          (seg[:, 0] == 2) * 1.0)
                    val_loss_list.append(loss.detach().cpu().numpy())
                    output_sigmoid = torch.sigmoid(output)
                    val_disc_dice_list.append(get_hard_dice(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                    val_cup_dice_list.append(get_hard_dice(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))
                    val_disc_IOU_list.append(get_hard_iou(output_sigmoid[:, 0].cpu(), (seg[:, 0] > 0).cpu() * 1.0))
                    val_cup_IOU_list.append(get_hard_iou(output_sigmoid[:, 1].cpu(), (seg[:, 0] == 2).cpu() * 1.0))


                    batch_end_time = time()
                    batch_time = batch_end_time - batch_start_time
                    time_per_image = batch_time / batch_size
                    val_time_per_image_cost_list.append(time_per_image)
                    val_memory_usage_list.append(torch.cuda.max_memory_allocated() / 1024 ** 2)

            mean_val_loss = np.mean(val_loss_list)
            mean_val_disc_dice = np.mean(val_disc_dice_list)
            mean_val_cup_dice = np.mean(val_cup_dice_list)
            mean_val_disc_IOU = np.mean(val_disc_IOU_list)
            mean_val_cup_IOU = np.mean(val_cup_IOU_list)

            # writer.add_scalar("Val Scalars/Val Loss", mean_val_loss, epoch)
            # writer.add_scalar("Val Scalars/Disc Dice", mean_val_disc_dice, epoch)
            # writer.add_scalar("Val Scalars/Cup Dice", mean_val_cup_dice, epoch)
            # writer.add_scalar("Val Scalars/Cup IOU", mean_val_cup_IOU, epoch)
            # writer.add_scalar("Val Scalars/Disc IOU", mean_val_disc_IOU, epoch)
            #
            # writer.add_image('Val/Input', make_grid(data[:10], 10, normalize=True), epoch)
            # writer.add_image('Val/Output Disc', make_grid(output_sigmoid[:10, 0][:, np.newaxis], 10, normalize=True), epoch)
            # writer.add_image('Val/Output Cup', make_grid(output_sigmoid[:10, 1][:, np.newaxis], 10, normalize=True), epoch)
            # writer.add_image('Val/Seg', make_grid(seg[:10], 10, normalize=True), epoch)

            print('  Val loss: {}\n'
                  '  Val disc dice: {}; Cup dice: {}\n'
                  '  Val disc IOU: {}; Cup IOU: {}\n'.format(mean_val_loss, mean_val_disc_dice, mean_val_cup_dice,
                                                             mean_val_disc_IOU, mean_val_cup_IOU))

        if epoch % save_interval == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, join(model_folder, 'model_latest.model'))
        if (epoch + 1) % 200 == 0:
            saved_model = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            print('  Saving model_{}.model...'.format(epoch + 1))
            torch.save(saved_model, join(model_folder, 'model_{}.model'.format(epoch + 1)))

        time_per_epoch = time() - start_epoch
        print('  Durations: {}'.format(time_per_epoch))
        # writer.add_scalar("Time/Time per epoch", time_per_epoch, epoch)

    saved_model = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    print('Saving model_{}.model...'.format('final'))
    torch.save(saved_model, join(model_folder, 'model_final.model'))
    if isfile(join(model_folder, 'model_latest.model')):
        os.remove(join(model_folder, 'model_latest.model'))
    total_time = time() - start
    print("Running %d epochs took a total of %.2f seconds." % (num_epochs, total_time))

    print(f'Average Time per image (train): {np.mean(time_per_image_cost_list):.4f} seconds')
    print(f"Average Max memory allocated (train): {np.mean(memory_usage_list):.2f} MB")
    print(f'Average Time per image (val): {np.mean(val_time_per_image_cost_list):.4f} seconds')
    print(f"Average Max memory allocated (val): {np.mean(val_memory_usage_list):.2f} MB")

    # inference
    from deeplearning.inference.inference_nets.inference_reins_RETFound_seg import inference
    for ts_csv_path in ts_csv:
        inference_tag = split_path(ts_csv_path)[-1].replace('.csv', '')
        print("Running inference: {}".format(inference_tag))
        inference('model_final.model', gpu, log_folder, patch_size, root_folder, [ts_csv_path], inference_tag)
