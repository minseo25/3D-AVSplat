import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import time
import csv
from tqdm import tqdm
from collections import OrderedDict
import soundfile as sf
import imageio.v2 as imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
import shutil

from config import init_args, params
# from data import *
import data
import models
from models import *
from utils import utils, torch_utils

from vis_functs import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def inference(args, pr, net, criterion, data_set, data_loader, device='cuda', video_idx=None):
    # import pdb; pdb.set_trace()
    net.eval()
    img_path_list = []
    crw_itds = []
    baseline_itds = []
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
            # import pdb; pdb.set_trace()
            img_paths = batch['img_path']
            audio = batch['audio']
            audio_rate = batch['audio_rate']
            delay_time = batch['delay_time'].to(device)
            out = predict(args, pr, net, batch, device)
            aff_L2R = criterion.inference(out, softmax=False)
            crw_itd = estimate_crw_itd(args, pr, aff_L2R, delay_time)
            crw_itds.append(crw_itd)
            for i in range(aff_L2R.shape[0]):
                img_path_list.append(img_paths[i])
                curr_audio = audio[i]

                if args.no_baseline:
                    baseline_itd = 0
                else:
                    baseline_itd = gcc_phat(args, pr, curr_audio, fs=audio_rate[i].item(), max_tau=pr.max_delay, interp=1)
                baseline_itds.append(baseline_itd)

    img_path_list = np.array(img_path_list)
    crw_itds = torch.cat(crw_itds, dim=-1).data.cpu().numpy() * 1000
    baseline_itds = np.array(baseline_itds) * 1000
    visualization(args, pr, data_set, data_loader, img_path_list, crw_itds, baseline_itds, video_idx)


def visualization(args, pr, data_set, data_loader, img_path_list, crw_itds, baseline_itds, video_idx):
    # Get video name from the first image path
    video_name = os.path.basename(os.path.dirname(os.path.dirname(img_path_list[0])))
    save_path = os.path.join('samples_tde', video_name)
    frame_folder = os.path.join(save_path, 'frames')
    audio_folder = os.path.join(save_path, 'audio')
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    # audio part
    audio = data_set.audio
    audio_rate = data_set.audio_rate
    audio_name = os.path.join(audio_folder, 'audio.wav')
    audio = audio.transpose(0, 1).data.cpu().numpy()
    sf.write(audio_name, audio, audio_rate)

    # frame part 
    frame_rate = args.frame_rate  # Use frame rate from command line argument
    for i in range(img_path_list.shape[0]):
        frame_name = os.path.join(frame_folder, f'frame-{str(i).zfill(3)}.jpg')
        frame_vis(args, pr, img_path_list, crw_itds, baseline_itds, frame_rate, i, frame_name)

    # Save time delay estimation results
    log_path = os.path.join(save_path, 'depth.log')
    with open(log_path, 'w') as f:
        f.write(f"# Frame rate: {frame_rate} fps\n")
        f.write(f"# Time chunk (s), Time delay (ms)\n")
        f.write(f"# Note: Negative values indicate left, positive values indicate right\n")
        for i in range(len(crw_itds)):
            start_time = i / frame_rate
            end_time = (i + 1) / frame_rate
            time_delay = -crw_itds[i]  # Negate the value to match visualization
            f.write(f"{start_time:.2f}-{end_time:.2f} {time_delay:.2f}\n")

    # Generate videos
    video_name = os.path.join(save_path, 'output.mp4')

    os.system(f"ffmpeg -v quiet -y -framerate {frame_rate} -i \"{frame_folder}\"/frame-%03d.jpg -i {audio_name} -vf \"crop=trunc(iw/2)*2:trunc(ih/2)*2\" -vcodec h264 -strict -2 -acodec aac -shortest {video_name}")
    os.system("rm -rf {}".format(audio_folder))
    os.system("rm -rf {}".format(frame_folder))




def frame_vis(args, pr, img_path_list, crw_itds, baseline_itds, frame_rate, i, frame_name):
    # import pdb; pdb.set_trace()
    sns.set_style('white')
    img_path = img_path_list[i]
    crw_itd = - crw_itds[: i+1]
    baseline_itd =  - baseline_itds[: i+1]

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, dpi=150) #, figsize=(12, 7)
    # create image
    img = imageio.v2.imread(img_path)
    ax0.imshow(img)
    ax0.axis('off')

    # create plot
    bins = np.arange(i+1) / frame_rate
    if args.no_baseline:
        res = [crw_itd]
    else:
        res = [crw_itd, baseline_itd]
    names = ['Ours', 'GCC-PHAT']
    markers=['o', '*']
    for idx in range(len(res)):
        ax1.plot(bins, res[idx], label=names[idx], marker=markers[idx], linewidth=1.5, markersize=2)
    
    full_bins = np.arange(img_path_list.shape[0]) / frame_rate
    ax1.plot(full_bins, [0] * img_path_list.shape[0], linewidth=1, markersize=0, linestyle='dashed', color='grey')

    ax1.legend(loc='upper center', ncol=2, frameon=False, fontsize=8)
    ax1.set_xlabel('Time (s)', fontsize=8)
    # ax1.set_xticks([])
    ax1.set_ylabel('Time delay (ms)', fontsize=8, rotation='vertical')
    ax1.set_xlim([0, int(img_path_list.shape[0] / frame_rate)])
    ax1.set_ylim([-1.0, 1.0])
    ax1.set_yticks([-1.0, 0, 1.0])
    ax1.tick_params(axis='x', labelsize=6)
    ax1.tick_params(axis='y', labelsize=6)
    ax1.margins(x=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('L                R', fontsize=8)
    ax2.set_ylim([-1.0, 1.0])
    ax2.set_yticks([-1.0, 0, 1.0])
    ax2.tick_params(axis='y', labelsize=6)

    plt.savefig(frame_name, bbox_inches='tight')
    plt.close()



def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    pr.dataloader = 'SingleVideoDataset'
    update_param(args, pr)
    # ----- make dirs for results ----- #
    os.makedirs('samples_tde', exist_ok=True)
    # ------------------------------------- #
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #

    # ----- Network ----- #
    net = models.__dict__[pr.net](args, pr, device=device).to(device)
    criterion = models.__dict__[pr.loss](args, pr, device)

    # -------- Loading checkpoints weights ------------- #
    if args.resume:
        resume = args.resume
        net, _ = torch_utils.load_model(resume, net, device=device, strict=False)

    # ------------------- #
    net = nn.DataParallel(net, device_ids=gpu_ids)

    if isinstance(pr.list_vis, str):
        samples = []
        csv_file = csv.DictReader(open(pr.list_vis, 'r'), delimiter=',')
        for row in csv_file:
            if not row in samples:
                samples.append(row)

    if args.max_sample > 0:
        samples = samples[: args.max_sample]
    
    #  --------- Testing ------------ #
    for i in tqdm(range(len(samples)), desc="Generating Video"):
        pr.list_test = samples[i]['path']
        print(pr.list_test)
        # ----- Dataset and Dataloader ----- #
        test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=False)
        # --------------------------------- #
        inference(args, pr, net, criterion, test_dataset, test_loader, device, video_idx=i)


if __name__ == '__main__':
    parser = init_args(return_parser=True)
    parser.add_argument('--list_vis', type=str, default=None, required=False)
    parser.add_argument('--frame_rate', type=float, default=4, help='Frame rate for video processing')
    args = parser.parse_args()
    test(args, DEVICE)