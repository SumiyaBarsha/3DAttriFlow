import argparse
from matplotlib import pyplot as plt
import torch
import munch
import yaml
from dataset_svr.trainer_dataset import build_dataset_val
import torch
from utils.train_utils import *
import logging
import importlib
import random
import munch
import yaml
import os
import argparse
from utils.model_utils import *
from tqdm import tqdm
from datetime import datetime

import open3d as o3d 

import warnings
warnings.filterwarnings("ignore")


def val():

     # Create a directory for saving the outputs
    output_dir = os.path.join('output_files_svr', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def visualize_feature_map(feature_map, num_channels=5, batch_idx = 0):
        """
        Visualize the feature maps for the first few channels of a CNN feature tensor.
        feature_map: shape (1, C, H, W) – we take the first image's feature map.
        """
        fm = feature_map[0].cpu().detach().numpy()  # (C, H, W) for the first image
        plt.figure(figsize=(15, 3))
        for j in range(min(num_channels, fm.shape[0])):
            plt.subplot(1, num_channels, j+1)
            plt.imshow(fm[j, :, :], cmap='viridis')
            plt.title(f"Channel {j}")
            plt.axis('off')
        # Save the visualization to file (since this is a script, not interactive)
        plt.savefig(os.path.join(output_dir, f"batch{batch_idx}_channel{i+1}.png"))
        plt.show()
    
    dataloader_test = build_dataset_val(args)

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
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
    
    ckpt = torch.load(args.load_model)
    net.module.load_state_dict(ckpt['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)

    net.module.eval()

    logging.info('Testing...')

    test_loss_l1 = AverageValueMeter()
    test_loss_l2 = AverageValueMeter()

    with tqdm(dataloader_test) as t:
        all_f1_scores = []
        for i, data in enumerate(t):
            with torch.no_grad():
        
                images = data['image'].cuda()
                gt = data['points'].cuda()

                batch_size = gt.shape[0]
                
                pred_points = net(images) # images shape (B, V, 3, H, W), model outputs shape (B, 2048, 3)

                loss_p, loss_t , f1 = calc_cd(pred_points, gt, calc_f1 = True)
                f1_mean = f1.mean().item()
                all_f1_scores.append(f1_mean)

                cd_l1_item = torch.sum(loss_p).item() / batch_size * 1e4
                cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
                test_loss_l1.update(cd_l1_item, batch_size)
                test_loss_l2.update(cd_l2_item, batch_size)
                
                # Update progress bar with current batch metrics
                t.set_postfix(CD_L1=cd_l1_item, CD_L2=cd_l2_item)

                # **Extract and visualize feature maps from the encoder**
                B, V, C, H, W = images.shape
                images_flat = images.view(B * V, C, H, W)
                encoder_output = net.module.encoder(images_flat)  
                if isinstance(encoder_output, tuple):
                    feat_vec, feat_map = encoder_output    # feat_map: mid-layer feature map
                else:
                    feat_vec, feat_map = encoder_output, None
                if feat_map is not None:
                    # Visualize feature maps for first sample’s first view
                    feature_map = feat_map[0:1, ...]  # shape (1, num_channels, H_f, W_f)
                    visualize_feature_map(feature_map, i)  # See function below

                # **Retrieve multi-view attention weights (self-attention)**
                feat_128 = net.module.linear(feat_vec).view(B, V, -1)            # project to 128-d
                attn_output, attn_weights = net.module.feature_fusion(feat_128)  # apply self-attention
                attn_weights = attn_weights.cpu().numpy()  # shape (B, num_heads, V, V)
                # Save attention weights for analysis
                np.save(os.path.join(output_dir, f"attn_weights_batch_{i}.npy"), attn_weights)
                
                # **Save outputs**: point cloud and reconstructed mesh for each sample
            for idx in range(pred_points.shape[0]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pred_points[idx].cpu().numpy())

                # Save PLY files (for later visualization)
                output_pc = os.path.join(output_dir, f"output_{i}_{idx}.ply")
                o3d.io.write_point_cloud(output_pc, pcd)

                # Save 2D point cloud visualization instead of Open3D's window
                visualize_point_cloud(pred_points[idx].cpu().numpy(), i, idx)

    print(f'cd_l1 {test_loss_l1.avg: .6f} cd_l2 {test_loss_l2.avg: .6f}')
    np.save(os.path.join(output_dir, "f1_scores.npy"), np.array(all_f1_scores))

def visualize_point_cloud(point_cloud, batch_idx, sample_idx):
    """Plots and saves a 2D view of a point cloud using Matplotlib."""
    output_dir = os.path.join("output_files_point_cloud", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')
    ax.set_title(f"Point Cloud (Batch {batch_idx}, Sample {sample_idx})")
    
    plt.savefig(os.path.join(output_dir, f"pointcloud_{batch_idx}_{sample_idx}.png"))
    plt.close()



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

    val()
