# Import required libraries
from torch.utils.data import DataLoader
from lossy_audio_classification_model.src.data_handling.dataset_classes import SpectrogramDataset, InferenceDataset


train_data_path = 'data/processed_data/train'
test_data_path = 'data/processed_data/test'


# Create datasets and dataloaders
train_dataset = SpectrogramDataset(train_data_path)
test_dataset = SpectrogramDataset(test_data_path)


train_loader = DataLoader(
    train_dataset, 
    batch_size=16,  # If memory-constrained
    shuffle=True,
    # num_workers = 1,  # Get number of CPU cores
    pin_memory=True,  # Faster GPU transfer
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,  
    shuffle=False,
    # num_workers = multiprocessing.cpu_count(),
    pin_memory=True,  
)