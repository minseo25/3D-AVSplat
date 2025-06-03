# 3D-AVSplat

## Installation

```bash
apt-get update
apt-get install ffmpeg python3.8-dev

# Create and activate virtual environment
python3.8 -m venv .venv
source .venv/bin/activate

# Install dependencies
export PIP_CACHE_DIR=/data/.cache/pip
pip install --cache-dir=$PIP_CACHE_DIR -r requirements.txt

# install detectron, mask2former
cd avis/avism/detectron2
pip install -e .
cd /data/3D-AVSplat/avis
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd /data/3D-AVSplat/

# download VGGISH checkpoints, params
cd audio_feature_extractor/
wget https://storage.googleapis.com/audioset/vggish_model.ckpt
wget https://storage.googleapis.com/audioset/vggish_pca_params.npz

# download AVIS checkpoints
# https://github.com/ruohaoguo/avis
cd avis/
mkdir pre_models # and download pre-trained backbones
mkdir checkpoints # and download model checkpoints
```

## Usage

1. Place your target video (must be mp4 format) in the `samples/` folder

2. Run the following command in the project root:
   ```bash
   python run.py -s your_video_name.mp4
   ```
   This will segment the video into chunks based on audio silence detection

3. The output will be organized as follows:
   ```   samples_chunked/
   └── video_name/
       ├── 0.25_5.25/
       │   ├── images/
       │   │   └── (image frames)
       │   ├── audio.wav
       │   ├── clap_audio_embeds.npy
       │   └── audio_feature.npy
       ├── 9.5_13.75/
       │   ├── images/
       │   │   └── (image frames)
       │   ├── audio.wav
       │   ├── clap_audio_embeds.npy
       │   └── audio_feature.npy
       └── 15.5_19.25/
           └── (same structure as above)
   ```

Each chunk folder contains:
- `images/`: Image frames used for AVIS (https://github.com/ruohaoguo/avis)
- `audio.wav`: Chunked audio file
- `clap_audio_embeds.npy`: CLAP audio embeddings
- `audio_feature.npy`: Audio features used for AVIS (https://github.com/ruohaoguo/avis)
