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
import tensorflow as tf
import wave
import sys
import torch.nn.functional as F

# 캐시 디렉토리 설정
CACHE_DIR = '/data/.cache'
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, 'huggingface'), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, 'torch/hub'), exist_ok=True)

# 환경 변수 설정
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['CLAP_CACHE_DIR'] = os.path.join(CACHE_DIR, 'clap')

# torch hub 캐시 디렉토리 설정
torch.hub.set_dir(os.path.join(CACHE_DIR, 'torch/hub'))

# add audio_feature_extractor to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio_feature_extractor'))
import vggish_input
import vggish_params
import vggish_slim

# 로깅 레벨 설정
logging.getLogger('laion_clap').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

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
        
        # 전체 길이를 chunk_length로 나눈 몫만큼만 처리
        total_chunks = wave.shape[1] // int(chunk_length * sr)
        wave = wave[:, :total_chunks * int(chunk_length * sr)]
        
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

def extract_vggish_features(audio_path, output_dir, step_sec=1.0):
    """
    Extract VGGish features from audio file
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save features
        step_sec: Time step in seconds (default: 1.0 for 1fps)
    """
    try:
        # VGGish model file path
        checkpoint_path = os.path.join('audio_feature_extractor', 'vggish_model.ckpt')
        pca_params_path = os.path.join('audio_feature_extractor', 'vggish_pca_params.npz')
        
        # calculate audio length
        with wave.open(audio_path, 'r') as f:
            total_samples = f.getnframes()
            sr = f.getframerate()
            duration_sec = total_samples / float(sr)
        
        # pass duration_sec and step_sec to split the audio
        input_batch = vggish_input.wavfile_to_examples(audio_path, duration_sec, step_sec=step_sec)
        
        # extract features from VGGish model
        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
            
            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})
            
            # save features
            output_path = os.path.join(output_dir, "audio_feature.npy")
            np.save(output_path, embedding_batch)
            print(f"Saved VGGish features to {output_path}")
            return True
    except Exception as e:
        print(f"Error in extract_vggish_features: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='SAT_T_1s')
    parser.add_argument('-c', '--chunk_length', type=float, default=0.25, choices=[0.2, 0.25, 0.5, 1.0])
    parser.add_argument('-s', '--sample_name', type=str, default='guitar.mp4')
    args = parser.parse_args()
    fps_lists = {0.2: 5, 0.25: 4, 0.5: 2, 1.0: 1}
    fps = fps_lists[args.chunk_length]

    # make directory for samples
    os.makedirs(SAMPLE_PATH, exist_ok=True)
    os.makedirs(SAMPLE_CHUNKED_PATH, exist_ok=True)
    
    # make directories for chunked data
    os.makedirs(os.path.join(SAMPLE_CHUNKED_PATH, "FEATAudios"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_CHUNKED_PATH, "FEATAudios_CLAP"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_CHUNKED_PATH, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_CHUNKED_PATH, "WAVAudios"), exist_ok=True)

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

    # process each non-silence segment
    target_name = args.sample_name.split(".")[0]
    for start, end in non_silence_segments:
        segment_name = f"{target_name}_{start}_{end}"
        
        # extract audio and save it to WAVAudios
        audio_path = os.path.join(SAMPLE_CHUNKED_PATH, "WAVAudios", f"{segment_name}.wav")
        if not os.path.exists(audio_path):
            command = ['ffmpeg', '-i', sample_video_path, '-ss', str(start), '-t', str(end - start), 
                      '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # pad audio
        wav_tensor, sr = torchaudio.load(audio_path)  # wav_tensor.shape = [1, orig_samples]
        orig_samples = wav_tensor.shape[1]
        target_samples = int((end - start) * sr)   # e.g. (9.0 - 1.0) * 16000 = 128000
        pad_samples = target_samples - orig_samples     # e.g. 128000 - 127464 = 536 (add padding to the end)

        if pad_samples > 0:
            # add padding to the end
            wav_tensor = F.pad(wav_tensor, (0, pad_samples))  # (left, right) tuple
            # save the padded audio
            torchaudio.save(audio_path, wav_tensor, sr)

        # extract image frames to JPEGImages
        image_dir = os.path.join(SAMPLE_CHUNKED_PATH, "JPEGImages", segment_name)
        os.makedirs(image_dir, exist_ok=True)
        command = ['ffmpeg', '-i', sample_video_path, '-ss', str(start), '-t', str(end - start), 
                  '-vf', f'fps={fps}', image_dir + '/%06d.jpg']
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # # print debugging information
        # print(f"\nSegment: {segment_name}")
        # print(f"Audio path: {audio_path}")
        # print(f"Image dir: {image_dir}")
        # frame_files = sorted(os.listdir(image_dir))
        # print(f"Number of frames: {len(frame_files)}")
        # print(f"Frame files: {frame_files}")
        # with wave.open(audio_path, 'r') as f:
        #     audio_frames = f.getnframes()
        #     audio_rate = f.getframerate()
        #     audio_duration = audio_frames / float(audio_rate)
        #     print(f"Audio duration: {audio_duration:.4f}s")
        #     print(f"Audio frames: {audio_frames}")
        #     print(f"Audio rate: {audio_rate}")
        # print(f"Expected duration (frames * chunk): {len(frame_files) * args.chunk_length:.4f}s")
        # print(f"Start: {start}, End: {end}, Duration: {end-start}")
        
        # extract and save VGGish features with step_sec
        vggish_output_path = os.path.join(SAMPLE_CHUNKED_PATH, "FEATAudios", f"{segment_name}.npy")
        if not os.path.exists(vggish_output_path):
            extract_vggish_features(audio_path, os.path.dirname(vggish_output_path), step_sec=args.chunk_length)
            # Rename the output file to match the desired structure
            temp_path = os.path.join(os.path.dirname(vggish_output_path), "audio_feature.npy")
            if os.path.exists(temp_path):
                os.rename(temp_path, vggish_output_path)
        
        # extract and save CLAP embeddings
        clap_output_path = os.path.join(SAMPLE_CHUNKED_PATH, "FEATAudios_CLAP", f"{segment_name}.npy")
        if not os.path.exists(clap_output_path):
            save_clap_embedding(audio_path, os.path.dirname(clap_output_path))
            # Rename the output file to match the desired structure
            temp_path = os.path.join(os.path.dirname(clap_output_path), "clap_audio_embeds.npy")
            if os.path.exists(temp_path):
                os.rename(temp_path, clap_output_path)

if __name__ == "__main__":
    main()