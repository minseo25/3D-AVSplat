# 3D-AVSplat

## Installation

```bash
# Install ffmpeg
apt-get install ffmpeg

# Create and activate virtual environment
python3.8 -m venv .venv
source .venv/bin/activate

# Install dependencies
export PIP_CACHE_DIR=/data/.cache/pip
pip install --cache-dir=$PIP_CACHE_DIR -r requirements.txt

# download checkpoints, params
cd audio_feature_extractor/
wget https://storage.googleapis.com/audioset/vggish_model.ckpt
wget https://storage.googleapis.com/audioset/vggish_pca_params.npz
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
