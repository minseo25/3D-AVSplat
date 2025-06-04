# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy
from scipy.io import wavfile

import mel_features
import vggish_params


def waveform_to_examples(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples


def wavfile_to_examples(wav_file, duration_sec, step_sec=1.0):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.
    Args:
        wav_file: String path to a file, or a file-like object (16-bit PCM WAV).
        duration_sec: Total duration in seconds (float).
        step_sec: Time step in seconds (e.g. 1.0 → 1fps, 0.5 → 2fps, 0.25 → 4fps).
    Returns:
        np.ndarray of shape (num_frames, 96, 64): log-mel spectrogram patches.
    """
    sr, snd = wavfile.read(wav_file)
    total_samples = int(round(sr * duration_sec))
    wav_data = snd[:total_samples]
    
    # Convert to mono if stereo
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    
    # calculate the actual number of frames (rounded up)
    num_frames = int(np.ceil(duration_sec / step_sec))
    log_mel = np.zeros([num_frames, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

    for i in range(num_frames):
        start_time = i * step_sec
        end_time = min((i + 1) * step_sec, duration_sec)
        duration = end_time - start_time
        
        # cut out the audio in sample unit
        start_sample = int(round(start_time * sr))
        end_sample = int(round(end_time * sr))
        data = wav_data[start_sample:end_sample]
        
        # if the last chunk is shorter than step_sec, pad with 0
        target_length = int(round(step_sec * sr))
        if data.shape[0] < target_length:
            data = np.pad(data, (0, target_length - data.shape[0]), mode='constant')

        wave_data_array = waveform_to_examples(data, sr)
        if wave_data_array.shape[0] > 0:
            log_mel[i, :, :] = wave_data_array[0]
        else:
            log_mel[i, :, :] = np.zeros((vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS))

    return log_mel
