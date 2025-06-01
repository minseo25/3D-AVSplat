import torch
import pandas as pd
import torchaudio
from pathlib import Path
import argparse
import audio_segmentation_models as models
import laion_clap
import os
import subprocess
import numpy as np
import logging

# 로깅 레벨 설정
logging.getLogger('laion_clap').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# cache directory setting for oom prevention
os.environ['TORCH_HOME'] = '/data/.cache'
torch.hub.set_dir('/data/.cache/torch/hub')
os.environ['HF_HOME'] = '/data/.cache/huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/huggingface'
os.environ['CLAP_CACHE_DIR'] = '/data/.cache/clap'

# parameters
DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
SR = 16000
SILENCE_IDX = 500
SAMPLE_PATH = "samples"
SAMPLE_CHUNKED_PATH = "samples_chunked"

# CLAP model for audio embedding
CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False)
CLAP_MODEL.load_ckpt()


def extract_audio_from_video(video_name):
    video_path = os.path.join(SAMPLE_PATH, video_name)
    audio_path = os.path.join(SAMPLE_PATH, video_name.replace(".mp4", ".wav"))

    if not os.path.exists(audio_path):
        # extract audio from the video
        command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path


def segment_audio(input_wav, model_name='SAT_T_1s', chunk_length=0.25):
    cl_lab_idxs_file = Path(__file__).parent / 'audio_segmentation/datasets/audioset/data/metadata/class_labels_indices.csv'

    label_maps = pd.read_csv(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        if not cl_lab_idxs_file.exists() else cl_lab_idxs_file).set_index(
            'index')['display_name'].to_dict()

    model = getattr(models, model_name)(pretrained=True)
    model = model.to(DEVICE).eval()

    non_silence_segments = []
    current_segment_start = None

    with torch.no_grad():
        zero_cache = None
        if 'SAT' in model_name:
            # First produce a "silence" cache
            *_, zero_cache = model(torch.zeros(
                1, int(model.cache_length / 100 * SR)).to(DEVICE),
                                   return_cache=True)

        wave, sr = torchaudio.load(input_wav)
        # if wave is not 16khz, resample to 16khz (SAT model is trained on 16khz)
        if sr != SR:
            resampler = torchaudio.transforms.Resample(sr, SR)
            wave = resampler(wave)
            sr = SR
        # if wave is stereo, convert to mono
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        assert sr == SR, "Models are trained on 16khz, please sample your input to 16khz"
        wave = wave.to(DEVICE)
        
        for chunk_idx, chunk in enumerate(wave.split(int(chunk_length * sr), -1)):
            if zero_cache is not None:
                output, zero_cache = model(chunk, cache=zero_cache, return_cache=True)
                output = output.squeeze(0)
            else:
                output = model(chunk).squeeze(0)
                                
            # get silence probability
            silence_prob = output[SILENCE_IDX].item()
            is_silence = silence_prob > 0.1
           
            current_time = chunk_idx * chunk_length
            
            if not is_silence:
                if current_segment_start is None:
                    current_segment_start = current_time
            else:
                if current_segment_start is not None:
                    non_silence_segments.append((current_segment_start, current_time))
                    current_segment_start = None
        
        # process the last segment
        if current_segment_start is not None:
            non_silence_segments.append((current_segment_start, (chunk_idx + 1) * chunk_length))
    
    return non_silence_segments


def save_clap_embedding(audio_path, output_dir):
    try:
        # extract embeddings from all wav files
        audio_embeds = CLAP_MODEL.get_audio_embedding_from_filelist(x=[str(audio_path)], use_tensor=False)
        print(f"Generated embeddings shape: {audio_embeds.shape}")
        
        # save embeddings to the output directory
        output_path = os.path.join(output_dir, "clap_audio_embeds.npy")
        np.save(output_path, audio_embeds)
        print(f"Saved embeddings to {output_path}")
        return True
    except Exception as e:
        print(f"Error in save_clap_embedding: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='SAT_T_1s')
    parser.add_argument('-c', '--chunk_length', type=float, default=0.25)
    parser.add_argument('-s', '--sample_name', type=str, default='funny_dogs.mp4')
    args = parser.parse_args()

    # check if the sample path exists
    sample_video_path = os.path.join(SAMPLE_PATH, args.sample_name)
    if not os.path.exists(sample_video_path):
        print(f"Warning: Sample path {sample_video_path} does not exist")
        return

    # extract audio from the video
    audio_path = extract_audio_from_video(args.sample_name)

    # segment audio into smaller chunks (non-silence segments)
    non_silence_segments = segment_audio(
        audio_path,
        model_name=args.model,
        chunk_length=args.chunk_length
    )

    # make output directory for chunked video
    target_name = args.sample_name.split(".")[0]
    base_output_dir = os.path.join(SAMPLE_CHUNKED_PATH, target_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # make embeddings for each non-silence segment
    for start, end in non_silence_segments:
        # make output directory for each segment
        segment_dir = os.path.join(base_output_dir, f"{start}_{end}")
        os.makedirs(segment_dir, exist_ok=True)

        # extract audio and save it to the output directory
        audio_path = os.path.join(segment_dir, f"{start}_{end}.wav")
        if not os.path.exists(audio_path):
            # extract audio from the video with precise timestamps
            command = ['ffmpeg', '-i', sample_video_path, '-ss', str(start), '-t', str(end - start), 
                      '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # extract embeddings and save it to the output directory
        save_clap_embedding(audio_path, segment_dir)
            

if __name__ == "__main__":
    main()