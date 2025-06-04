import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import multiprocessing as mp

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'avis'))
# fmt: on

import tempfile
import time
import cv2
import numpy as np
import re

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from avism import add_avism_config
from predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_avism_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="avism demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="avis/configs/avism/R50/avism_R50_IN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "avis/checkpoints/AVISM_R50_IN.pth"],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--input-dir",
        default="avis/datasets/test/JPEGImages/",
        help="Directory containing input video frames",
    )
    parser.add_argument(
        "--output-dir",
        default="avis/results/",
        help="Directory to save output visualizations",
    )
    parser.add_argument(
        "--audio-dir",
        default="avis/datasets/test/FEATAudios/",
        help="Directory containing audio features",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


def extract_number(filename):
    return int(re.search(r'(\d+).jpg$', filename).group(1))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for video_name in os.listdir(args.input_dir):
        print(video_name)
        vid_frames = []
        for path in sorted(os.listdir(os.path.join(args.input_dir, video_name)), key=extract_number):
            img = read_image(os.path.join(args.input_dir, video_name, path), format="BGR")
            vid_frames.append(img)

        audio_pth = os.path.join(args.audio_dir, video_name + ".npy")
        audio_feats = np.load(audio_pth)

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames, audio_feats)

        os.makedirs(os.path.join(args.output_dir, video_name), exist_ok=True)

        for path, _vis_output in zip(sorted(os.listdir(os.path.join(args.input_dir, video_name)), key=extract_number), visualized_output):
            out_filename = os.path.join(args.output_dir, video_name, path)
            _vis_output.save(out_filename)
