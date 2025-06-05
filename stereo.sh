#!/bin/bash
frame_rate=${1:-4}

echo "Name of the video folder: TEST"
echo "Frame rate: $frame_rate"

# Create necessary directories
mkdir -p stereocrw/preprocess/RawVideos/TEST
mkdir -p stereocrw/preprocess/ProcessedData/TEST
mkdir -p stereocrw/preprocess/data-split/TEST
mkdir -p samples_tde

# Copy video files
cp samples/*.mp4 stereocrw/preprocess/RawVideos/TEST/

# Process videos
python stereocrw/preprocess/process.py --split=0 --total=1 --dataset_name=TEST --frame_rate=$frame_rate 
python stereocrw/preprocess/create-csv.py --dataset_name=TEST --type='' --data_split='1:0:0' --unshuffle

# Generate Visualization Results
{
CUDA=0
Model='stereocrw/checkpoints/pretrained-models/FreeMusic-StereoCRW-1024.pth.tar'

clip=0.24
patchsize=3840
patchstride=1
patchnum=512
mode='mean'

echo 'Generating Visualization Results......'
CUDA_VISIBLE_DEVICES=$CUDA python stereocrw/vis_scripts/vis_video_itd.py --exp=TEST --setting='stereocrw_binaural' --backbone='resnet9' --batch_size=2 --num_workers=8 --max_sample=-1 --resume=$Model --patch_stride=$patchstride --patch_num=$patchnum --clip_length=$clip --wav2spec --mode=$mode  --gcc_fft=$patchsize --list_vis=stereocrw/preprocess/data-split/TEST/vis.csv --no_baseline --frame_rate=$frame_rate
}

# Delete temporary files in the end
rm -rf stereocrw/preprocess/RawVideos/
rm -rf stereocrw/preprocess/ProcessedData/
rm -rf stereocrw/preprocess/data-split/