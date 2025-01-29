import pandas as pd
import torch
from nnAudio import features as spectrograms
import io
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Helper functions - Using device: {device}')

# Clear memory cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def unzip_to_buffer(zip_ref, track):
    """Open the zipfolder and save file-like object to memory."""
    waveform_binary = zip_ref.read(track)
    file_like_object = io.BytesIO(waveform_binary)

    return file_like_object

def generate_spectrograms(waveforms):
    """Generate spectrograms from a batch of waveforms.
    
    Args:
        waveforms: numpy array of shape (batch_size, samples) or torch tensor
    Returns:
        batch of spectrograms of shape (batch_size, freq_bins, time_frames)
    """
    # If input is a single waveform, add batch dimension
    if len(waveforms.shape) == 1:
        waveforms = waveforms[None, :]
        
    # Convert to tensor if not already
    if not torch.is_tensor(waveforms):
        waveforms = torch.tensor(waveforms)
    
    waveforms = waveforms.to(device)  # Move batch to device
    spec_converter = spectrograms.STFT(n_fft=2048).to(device)
    spectrograms_batch = spec_converter(waveforms)
    spectrograms_batch = spectrograms_batch.cpu().detach().numpy()
    spectrograms_db = librosa.amplitude_to_db(np.abs(spectrograms_batch), ref=np.max)
    
    return spectrograms_db[:, :, :, 0]  # Return all spectrograms, keeping batch dimension

def save_track_data(track_name, spectrograms, path, category, is_training):
    """Save track data to disk."""
    if is_training:
        save_dir = os.path.join(path, 'train', category)
    else:
        save_dir = os.path.join(path, 'test', category)
    os.makedirs(save_dir, exist_ok=True)
    
    for segment_idx, spectrogram in enumerate(spectrograms):
        audio_file_row = pd.DataFrame([{
            'file_name': f'{track_name}_segment_{segment_idx}',
            'spectrogram': spectrogram
        }])
        audio_file_row.to_pickle(os.path.join(save_dir, f'{track_name}_segment_{segment_idx}.pkl'))

def is_training_track(track_components):
    """Check if track is valid for processing."""
    if track_components[0] == "train":
        return True
    else:
        return False

def create_lossless_waveform(wav_buffer):
    """Create lossless waveform from wav buffer."""
    wav_buffer.seek(0)
    waveform = librosa.load(wav_buffer, sr=None)
    return waveform

def create_lossy_waveform(wav_buffer):
    """Create lossy waveform from wav buffer."""
    wav_buffer.seek(0)
    mp3_buffer = io.BytesIO()
    AudioSegment.from_wav(wav_buffer).export(mp3_buffer, format="mp3", bitrate="320k")
    mp3_buffer.seek(0)
    waveform = librosa.load(mp3_buffer, sr=None)
    return waveform

def split_waveform(waveform, sr, segment_duration=6):
    """Split waveform into segments of specified duration.
    Returns numpy array of shape (num_segments, samples_per_segment)"""
    samples_per_segment = sr * segment_duration
    num_segments = int(np.ceil(len(waveform) / samples_per_segment))
    segments = np.zeros((num_segments, samples_per_segment), dtype=np.float32)
    print('\rnum_segments   | ', num_segments, end='', flush=True)
    for i in range(num_segments):
        start = i * samples_per_segment
        end = min(start + samples_per_segment, len(waveform))
        segment = waveform[start:end]
        
        # Pad last segment if needed
        if len(segment) < samples_per_segment:
            padding = np.zeros(samples_per_segment - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, padding])
            
        segments[i] = segment
    
    return segments

def process_and_save_track(zip_ref, track, track_name, path, is_training):
    """Process a single track to generate and save spectrograms."""
    wav_buffer = unzip_to_buffer(zip_ref, track)
    
    lossless_waveform, lossless_sr = create_lossless_waveform(wav_buffer)
    lossy_waveform, lossy_sr = create_lossy_waveform(wav_buffer)
    
    # Split waveforms into 30-second segments
    lossless_segments = split_waveform(lossless_waveform, lossless_sr)
    lossy_segments = split_waveform(lossy_waveform, lossy_sr)

    lossless_spectrograms = generate_spectrograms(lossless_segments)
    lossy_spectrograms = generate_spectrograms(lossy_segments)

    save_track_data(track_name, lossless_spectrograms, path, 'lossless', is_training)
    save_track_data(track_name, lossy_spectrograms, path, 'lossy', is_training)

def is_suitable_file(components):
    if components[-1] == 'mixture.wav':
        return True
    else:
        return False