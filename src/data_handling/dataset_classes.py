# Import required libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd

train_data_path = 'data/processed_data/train'
test_data_path = 'data/processed_data/test'

# Custom dataset class for train / test spectrograms
class SpectrogramDataset(Dataset):
    def __init__(self, path, target_size=(512, 512)):
        self.path = path
        self.samples = []
        self.target_size = target_size

        # Load all spectrogram files and their labels
        for catergory in ['lossy', 'lossless']:
            catergory_path = os.path.join(path, catergory)
            if not os.path.exists(catergory_path):
                raise RuntimeError(f"Directory not found: {catergory_path}")
            for file in os.listdir(catergory_path):
                if file.endswith('.pkl'):
                    self.samples.append({
                        'path': os.path.join(catergory_path, file),
                        'label': 0 if catergory == 'lossless' else 1
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def resize_spectrogram(self, spec):
        # print('\rresizing spectrogram from ', spec.shape, 'to ', self.target_size, end='', flush=True)
        
        # Add batch and channel dimensions before resizing
        spec = spec.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
        
        resize_transform = transforms.Compose([
            transforms.Resize(self.target_size)
        ])
        
        spec = resize_transform(spec)
        spec = spec.squeeze(0).squeeze(0)  # Remove the batch and channel dimensions
        return spec
    
    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            df = pd.read_pickle(sample['path'])
            
            spectrogram = df['spectrogram'].iloc[0]
            # print(f"\nLoading sample {idx}")
            # print(f"Raw spectrogram shape: {spectrogram.shape}")
            
            # Normalize spectrogram
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
            
            # Convert to tensor and pad/crop
            spectrogram = torch.FloatTensor(spectrogram)
            spectrogram = self.resize_spectrogram(spectrogram)
            spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension
            # print(f"Final tensor shape (with channel dimension): {spectrogram.shape}")
            
            return spectrogram, sample['label']
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            raise


# Create inference dataset for single spectrograms through the API. Dataloaders are declared in the app.py
class InferenceDataset(Dataset):
    def __init__(self, spectrograms):
        self.spectrograms = spectrograms
        self.target_size = (512, 512)  # Match training size
        
    def __len__(self):
        return len(self.spectrograms)
    
    def resize_spectrogram(self, spec):
        # print('\rresizing spectrogram from ', spec.shape, 'to ', self.target_size, end='', flush=True)
        
        # Add batch and channel dimensions before resizing
        spec = spec.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
        
        resize_transform = transforms.Compose([
            transforms.Resize(self.target_size)
        ])
        
        spec = resize_transform(spec)
        spec = spec.squeeze(0).squeeze(0)  # Remove the batch and channel dimensions
        return spec
    
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = torch.FloatTensor(spec)
        spec = self.resize_spectrogram(spec)
        spec = spec.unsqueeze(0)
        
        return spec