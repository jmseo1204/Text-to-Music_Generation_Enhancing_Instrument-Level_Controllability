import os
import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import AudioLDM2Pipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
import librosa
import random
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import scipy
import gc


# import torch_xla
# import torch_xla.core.xla_model as xm
def waveform_to_mel_spectrogram(audio_path):
    transform_sr = 16000  # pipeline.vocoder.config.sampling_rate
    n_mels = 64  # pipeline.vocoder.config.model_in_dim
    hop_length_ms = 10
    hop_length = int(transform_sr * hop_length_ms / 1000)  # must be 160 (equals to sr=16000, hop_length_ms=10)
    frame_length = 1024  # domain tradition
    n_fft = frame_length
    duration = 10.24

    y, _ = librosa.load(audio_path, sr=transform_sr, duration=duration)
    y = audio_volumn_regularization(y)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=transform_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        n_mels=n_mels,
    )
    print(f"mel spec length: {mel_spectrogram.shape[1]}")

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0)

    return torch.tensor(log_mel_spectrogram.T).unsqueeze(0)


def waveform_to_latent(audio_path, pipeline):
    mel = waveform_to_mel_spectrogram(audio_path).unsqueeze(0).to(device)
    latent = pipeline.vae.encode(mel).latent_dist.sample()
    latent = latent * pipeline.vae.config.scaling_factor
    return latent


def latent_to_waveform_for_test(latent_path, pipeline, audio_save_path=None):
    latent = torch.load(latent_path).to(device)
    scaled_latent = 1 / pipeline.vae.config.scaling_factor * latent
    mel_spectrogram = pipeline.vae.decode(scaled_latent).sample

    vocoder_dtype = next(pipeline.vocoder.parameters()).dtype
    mel_spectrogram = mel_spectrogram.to(pipeline.device, dtype=vocoder_dtype)

    audio = pipeline.mel_spectrogram_to_waveform(mel_spectrogram).detach().cpu().numpy()[0]
    audio_save_path = "reconstructed_audio_from_latent.wav" if audio_save_path is None else audio_save_path
    scipy.io.wavfile.write(audio_save_path, rate=16000, data=audio)


def audio_volumn_regularization(audio):
    basic_rms = 0.15
    rms = np.sqrt(np.mean(np.square(audio)))
    print(f"before audio regularizing(RMS): {rms}")
    audio *= basic_rms / rms
    return audio


def mix_audios(audio1, audio2, weight1=1.0):
    # audio1, audio2 의 결합 비율 = weight1 : 1

    len1, len2 = len(audio1), len(audio2)
    if len1 > len2:
        audio2 = np.pad(audio2, (0, len1 - len2), mode="constant")
    elif len1 < len2:
        audio1 = np.pad(audio1, (0, len2 - len1), mode="constant")

    calculate_rms = lambda x: np.sqrt(np.mean(np.square(x)))
    rms1 = calculate_rms(audio1)
    rms2 = calculate_rms(audio2)

    scaled_audio1 = weight1 / (weight1 + 1) * audio1
    scaled_audio2 = audio2 * rms1 / rms2 / (weight1 + 1)
    combined_audio = scaled_audio1 + scaled_audio2

    max_val = np.max(np.abs(combined_audio))
    if max_val > 1.0:
        combined_audio = combined_audio / max_val  # Peak normalization

    combined_audio = audio_volumn_regularization(combined_audio)

    return combined_audio

    # mixed_audio = amplify*(weight1 * audio1 + (1-weight1) * audio2)

    # return mixed_audio


def mix_with_original(inst, music_type, guidance=None, weight1=1.0, inst_music_path=None, background_music_path=None):
    duration = 10.24
    sr = 16000
    if inst_music_path is None:
        inst_music_path = f"[{inst}]{music_type}_with_guidance_{guidance}.wav"
    inst_music, _ = librosa.load(inst_music_path, sr=sr, duration=duration)

    background_music_path = music_type + "_as_reference.wav" if background_music_path is None else background_music_path
    original_music, _ = librosa.load(background_music_path, sr=sr, duration=duration)
    mixed_file_path = (
        f"[{inst}_mixed_with_original]{music_type}"
        + (f"_with_guidance_{guidance}" if guidance is not None else "")
        + ".wav"
    )
    scipy.io.wavfile.write(mixed_file_path, rate=sr, data=mix_audios(original_music, inst_music, weight1=weight1))
    return mixed_file_path


def mix_two_inst(inst1, inst2, music_type, guidance1, guidance2=None, weight1=1.0):
    sr = 16000
    duration = 10.24
    guidance2 = guidance1 if guidance2 is None else guidance2

    inst1_music_path = f"[{inst1}]{music_type}_with_guidance_{guidance1}.wav"
    inst1_music, _ = librosa.load(inst1_music_path, sr=sr, duration=duration)

    inst2_music_path = f"[{inst2}]{music_type}_with_guidance_{guidance2}.wav"
    inst2_music, _ = librosa.load(inst2_music_path, sr=sr, duration=duration)

    mixed_file_path = f"[{inst1}_and_{inst2}]{music_type}.wav"
    scipy.io.wavfile.write(mixed_file_path, rate=16000, data=mix_audios(inst1_music, inst2_music))
    return mixed_file_path


def make_latent_from_original(music_type):
    music_latent = waveform_to_latent(f"{music_type}_as_reference.wav", pipeline)
    latent_path = music_type + "_latent_tensor.pt"
    torch.save(music_latent, latent_path)
    latent_to_waveform_for_test(latent_path, pipeline)
    return latent_path
