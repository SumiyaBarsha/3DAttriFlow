import argparse
import logging
import os
import munch
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import logging
import torch
from tqdm import tqdm
from dataset_pc.dataset import MVP_CP
from utils.model_utils import calc_cd
import argparse
import torch
import logging
import lib
import random
import munch
import yaml
import os
import sys
import argparse
from dataset_pc.dataset import MVP_CP
from tqdm import tqdm
from time import time
import time as timetmp

import open3d as o3d
from dateutil import tz
from datetime import datetime

from utils.model_utils import *
from utils.train_utils import AverageValueMeter

def val():

    # Directory for output files
    current_time = datetime.now(tz=tz.tzlocal())
    output_dir = os.path.join('output_files', current_time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it does not exist
        
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    test_dataset_loader = MVP_CP(prefix="test")
    test_data_loader = torch.utils.data.DataLoader(test_dataset_loader, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)
    logging.info('Length of test dataset:%d', len(test_dataset_loader))

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

    # Switch models to evaluation mode
    net.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    test_losses = AverageValueMeter()

    with tqdm(test_data_loader) as t:
        for model_idx, data in enumerate(t):

            with torch.no_grad():

                category, label, partial, gt = data
                partial = partial.float().cuda()  # B x 2048 x 3 (incomplete input)
                gt = gt.float().cuda()  # B x 2048 x 3 (ground truth complete cloud)
                label = label.float().cuda()  # B x num_classes (one-hot label vector)

                ## Forward pass: get predicted complete point cloud from the model
                #pred_points = net(partial, label)       # Model now returns only the points (B, N, 3)
                ## Ensure shape is (B, N, 3) for Chamfer Distance calculation
                #if pred_points.shape[1] == 3:  
                #    pred_points = pred_points.transpose(2, 1).contiguous()  # transpose to (B, N, 3) if needed
                
                # **Use validation mode to get coarse and fine outputs**
                result = net.module(partial, gt, label, prefix="val")  
                coarse_points = result['out1']    # coarse output (B, N, 3)
                final_points = result['out2']     # final output (B, N, 3)
                cd_t = result['cd_t'].mean().item() * 1e4  # Chamfer L2 loss for this batch

                test_losses.update(cd_t, partial.shape[0])

                t.set_description('Test[%d/%d] ChamferL2 = %.4f' %
                             (model_idx + 1, n_samples, cd_t))

                for idx in range(coarse_points.shape[0]):
                    # Convert to numpy
                    coarse_np = coarse_points[idx].cpu().numpy()
                    fine_np = final_points[idx].cpu().numpy()
                
                    # Create Open3D point clouds
                    pcd_coarse = o3d.geometry.PointCloud()
                    pcd_coarse.points = o3d.utility.Vector3dVector(coarse_np)
                    pcd_coarse.paint_uniform_color([1.0, 0.0, 0.0])  # Red (coarse)
                
                    pcd_final = o3d.geometry.PointCloud()
                    pcd_final.points = o3d.utility.Vector3dVector(fine_np)
                    pcd_final.paint_uniform_color([0.0, 1.0, 0.0])  # Green (refined)
                
                    # Save PLY files (optional, can be downloaded from Kaggle)
                    o3d.io.write_point_cloud(os.path.join(output_dir, f"output_coarse_{model_idx}_{idx}.ply"), pcd_coarse)
                    o3d.io.write_point_cloud(os.path.join(output_dir, f"output_final_{model_idx}_{idx}.ply"), pcd_final)
                
                    # Convert Open3D visualization to Matplotlib-friendly format
                    fig = plt.figure(figsize=(10, 5))
                
                    # First subplot - Coarse point cloud
                    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                    ax1.scatter(coarse_np[:, 0], coarse_np[:, 1], coarse_np[:, 2], c='red', marker='o')
                    ax1.set_title("Coarse Point Cloud")
                
                    # Second subplot - Final point cloud
                    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                    ax2.scatter(fine_np[:, 0], fine_np[:, 1], fine_np[:, 2], c='green', marker='o')
                    ax2.set_title("Refined Point Cloud")
                
                    plt.savefig(os.path.join(output_dir, f"evolution_{model_idx}_{idx}.png"))  # Save figure
                    plt.show()  # Display in Kaggle Notebook

    print('============================ TEST RESULTS ============================')

    print('Overall cd_L2 (scaled): ', (test_losses.avg))



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
