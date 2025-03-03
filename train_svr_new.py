import argparse
import torch
import munch
import yaml
from dataset_svr.trainer_dataset_new import build_dataset
import torch.optim as optim
from models.PC.utils import fps_subsample
from utils.train_utils import *
import logging
import math
import importlib
import random
import os
import sys
import argparse
from tqdm import tqdm
from time import time
import time as timetmp
from utils.model_utils import *
# Imports for visualization
import matplotlib.pyplot as plt
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


import warnings
warnings.filterwarnings("ignore")


def setFolders(args):

    LOG_DIR = args.dir_outpath
    MODEL_NAME = '%s-%s'%(args.model_name, timetmp.strftime("%m%d_%H%M", timetmp.localtime()))

    OUT_DIR = os.path.join(LOG_DIR, MODEL_NAME)
    args.dir_checkpoints = os.path.join(OUT_DIR, 'checkpoints')
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    if not os.path.exists(args.dir_checkpoints):
        os.makedirs(args.dir_checkpoints)


    os.system('cp -r models %s' % (OUT_DIR))
    os.system('cp train_svr.py %s' % (OUT_DIR))

    os.system('cp -r cfgs %s' % (OUT_DIR))

    LOG_FOUT = open(os.path.join(OUT_DIR, 'log_%s.csv' %(MODEL_NAME)), 'w')
    return MODEL_NAME, OUT_DIR, LOG_FOUT

def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def visualize_feature_map(feature_map, num_channels=5):
    """
    Visualize the feature maps for the first few channels of a CNN feature tensor.
    feature_map: shape (1, C, H, W) – we take the first image's feature map.
    """
    output_dir = os.path.join("output_files_ft_map", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy and transpose for visualization purposes (C, H, W) -> (H, W, C)
    feature_map = feature_map.permute(1, 2, 3, 0).contiguous().cpu().detach().numpy()
    fm = feature_map[0].cpu().detach().numpy()  # (C, H, W) for the first image
    plt.figure(figsize=(15, 3))
    for j in range(min(num_channels, fm.shape[0])):
        plt.subplot(1, num_channels, j+1)
        plt.imshow(fm[j, :, :], cmap='viridis')
        plt.title(f"Channel {j}")
        plt.axis('off')
    # Save the visualization to file (since this is a script, not interactive)
    plt.savefig(os.path.join(output_dir, f'feature_map_visualization.png'))
    plt.show()

def train():
    # Set up folders for logs and checkpoints
    exp_name, log_dir, LOG_FOUT = setFolders(args)

    log_string('EPOCH,CD_L1,BEST CDL1,CD_L2,BEST CDL2', LOG_FOUT)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
                                                      
    logging.info(str(args))

    # Initialize TensorBoard writer #newly added
    writer = SummaryWriter(log_dir=log_dir + '/tensorboard_logs')

    metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}
    # Initialize lists for logging losses #newly added
    train_losses = []
    val_losses = []

    dataloader, dataloader_test = build_dataset(args)

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()

    print('# encoder parameters:', sum(param.numel() for param in net.module.encoder.parameters()))
    print('# decoder parameters:', sum(param.numel() for param in net.module.decoder.parameters()))

    lr = args.lr

    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))
    optimizer = optimizer(filter(lambda p: p.requires_grad,net.module.parameters()), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    best_cd_l1 = float("inf")
    best_cd_l2 = float("inf")

    for epoch in range(args.start_epoch, args.nepoch):
        epoch_start_time = time()
        total_cd_l1 = 0
        total_cd_l2 = 0

        train_loss_meter.reset()
        net.module.train()

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        n_batches = len(dataloader)
        with tqdm(dataloader) as t:
            for batch_idx, data in enumerate(t):
                optimizer.zero_grad()

                images = data['image'].cuda()
                gt = data['points'].cuda()

                batch_size = gt.shape[0]

                pred_points = net(images)

                net_loss, loss_t = calc_cd(pred_points, gt)
                reg_loss = net.module.decoder.orthogonal_regularizer() * 100
                net_loss = net_loss.mean()
                net_loss_all = net_loss + reg_loss

                train_loss_meter.update(net_loss.item())
                net_loss_all.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
                optimizer.step()

                cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
                total_cd_l2 += cd_l2_item
                cd_l1_item = net_loss.item() * 1e4
                total_cd_l1 += cd_l1_item

                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch, args.nepoch, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_l1_item, cd_l2_item]])
        
        avg_cd_l1 = total_cd_l1 / n_batches
        avg_cd_l2 = total_cd_l2 / n_batches
        # Log training loss to TensorBoard #newly added
        writer.add_scalar('Loss/Train', avg_cd_l1, epoch)

        # Save losses for visualization later #newly added
        train_losses.append(avg_cd_l1)
        with open(os.path.join(log_dir, 'train_losses.json'), 'w') as f:
            json.dump(train_losses, f)

        
        epoch_end_time = time()
        logging.info(' ')
        logging.info(
            exp_name + '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch, args.nepoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cd_l1, avg_cd_l2]]))

        if epoch % args.epoch_interval_to_save == 0:
            save_model(str(log_dir) + '/checkpoints/' + str(epoch) + 'network.pth', net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_cd_l1, best_cd_l2 = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, LOG_FOUT, log_dir, best_cd_l1, best_cd_l2)
            val_losses.append(best_cd_l1)  # Save validation loss #newly added
            with open(os.path.join(log_dir, 'val_losses.json'), 'w') as f:
                json.dump(val_losses, f)
            writer.add_scalar('Loss/Validation', best_cd_l1, epoch)  # Log validation loss to TensorBoard  #newly added
    writer.close()  # Close TensorBoard writer #newly added
    plot_loss_curve(log_dir)  # Plot loss curve after training #newly added
    # **Feature map visualization after training** (using final model on a sample)
    sample_data = next(iter(dataloader_test))
    sample_img = sample_data['image'][0:1].cuda()  # one sample (1, V, 3, H, W)
    with torch.no_grad():
        # Extract encoder feature map for sample image(s)
        flat_img = sample_img.view(-1, 3, sample_img.size(-2), sample_img.size(-1))
        enc_out = net.module.encoder(flat_img)
        if isinstance(enc_out, tuple):
            _, feat_map = enc_out
        else:
            feat_map = None
        if feat_map is not None:
            np.save(os.path.join(log_dir, "final_feature_map.npy"), feat_map.cpu().numpy())  # save features
            visualize_feature_map(feat_map[0:1, ...])  # plot and save the feature maps of first sample


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, LOG_FOUT, log_dir, best_cd_l1, best_cd_l2):
    
    val_start_time = time()
    metrics_val = ['cd_t']
    val_loss_meters = {m: AverageValueMeter() for m in metrics_val}
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    total_cd_l1 = 0
    total_cd_l2 = 0
    n_batches = len(dataloader_test) 
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            images = data['image'].cuda()
            gt = data['points'].cuda()

            batch_size = gt.shape[0]

            pred_points = net(images)

            loss_p, loss_t = calc_cd(pred_points, gt)

            cd_l1_item = torch.sum(loss_p).item() / batch_size * 1e4
            cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
            total_cd_l1 += cd_l1_item
            total_cd_l2 += cd_l2_item
        
        avg_cd_l1 = total_cd_l1 / n_batches
        avg_cd_l2 = total_cd_l2 / n_batches

        if avg_cd_l1 < best_cd_l1:
            best_cd_l1 = avg_cd_l1
            save_model(str(log_dir) + '/checkpoints/bestl1_network.pth', net)
            logging.info("Saving net...")

        if avg_cd_l2 < best_cd_l2:
            best_cd_l2 = avg_cd_l2
            save_model(str(log_dir) + '/checkpoints/bestl2_network.pth', net)
            logging.info("Saving net...")

        log_string('%d,%.2f,%.2f,%.2f,%.2f'%(curr_epoch_num, avg_cd_l1, best_cd_l1, avg_cd_l2, best_cd_l2), LOG_FOUT)

        val_end_time = time()

        logging.info(
            '[Epoch %d/%d] TestTime = %.3f (s) Curr_cdl1 = %s Best_cdl1 = %s Curr_cdl2 = %s Best_cdl2 = %s' %
            (curr_epoch_num, args.nepoch, val_end_time - val_start_time, avg_cd_l1, best_cd_l1, avg_cd_l2, best_cd_l2))        

    return best_cd_l1, best_cd_l2


#newly added
def plot_loss_curve(log_dir):
    with open(os.path.join(log_dir, 'train_losses.json'), 'r') as f:
        train_losses = json.load(f)
    val_losses_path = os.path.join(log_dir, 'val_losses.json')
    if os.path.exists(val_losses_path):
        with open(val_losses_path, 'r') as f:
            val_losses = json.load(f)
    else:
        val_losses = []  # Use an empty list if the file doesn't exist
        print("Warning: val_losses.json not found. Skipping validation loss plot.")

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.show()
#done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    print('Using gpu:' + str(arg.gpu_id))

    train()

