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
        default="avis/configs/avism/R50/avism_R50_COCO.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "avis/checkpoints/AVISM_R50_COCO.pth"],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--input-dir",
        default="samples_chunked/JPEGImages", # "avis/datasets/test/JPEGImages/",
        help="Directory containing input video frames",
    )
    parser.add_argument(
        "--output-dir",
        default="samples_avis", # "avis/results/",
        help="Directory to save output visualizations",
    )
    parser.add_argument(
        "--audio-dir",
        default="samples_chunked/FEATAudios", # "avis/datasets/test/FEATAudios/",
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

# given _mask with given image, draw a white and black mask
def draw_binary_mask(image, _mask):
    mask_img = (_mask.astype(np.uint8)) * 255  # 0 or 255
    mask_img = np.stack([mask_img]*3, axis=-1)
    return mask_img

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
            predictions, visualized_output, binary_mask_output = demo.run_on_video(vid_frames, audio_feats)

        os.makedirs(os.path.join(args.output_dir, video_name), exist_ok=True)

        for path, _vis_output, _mask in zip(sorted(os.listdir(os.path.join(args.input_dir, video_name)), key=extract_number), visualized_output, binary_mask_output):
            out_filename = os.path.join(args.output_dir, video_name, path)
            _vis_output.save(out_filename)

            # Save mask for npy file an binary image
            mask_base = os.path.splitext(path)[0] + ".npy"
            mask_save_dir = os.path.join(args.output_dir, video_name, "masks")
            os.makedirs(mask_save_dir, exist_ok=True)
            mask_filename = os.path.join(mask_save_dir, mask_base)
            np.save(mask_filename, _mask)

            # Save mask as black & white image (png)
            binary_image_dir = os.path.join(args.output_dir, video_name, "binary_masks")
            os.makedirs(binary_image_dir, exist_ok=True)
            if _mask is not None:
                mask_img = draw_binary_mask(vid_frames[extract_number(path)-1], _mask)
                mask_img_filename = os.path.join(binary_image_dir, os.path.splitext(path)[0] + ".png")
                cv2.imwrite(mask_img_filename, mask_img)
            else:
                print(f"No mask for {path}, skipping binary mask saving.")
