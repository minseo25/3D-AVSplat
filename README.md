Installation

```bash
apt-get install ffmpeg

python3.8 -m venv .venv
source .venv/bin/activate
export PIP_CACHE_DIR=/data/.cache/pip
pip install --cache-dir=$PIP_CACHE_DIR -r requirements.txt
```