# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import os

# import objects
from src.models.neural_networks.cnn import AudioClassifier
from src.data_handling.data_loaders import train_loader, test_loader



train_data_path = 'data/processed_data/train'
test_data_path = 'data/processed_data/test'

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
max_epochs = 20  # Set a maximum number of epochs
patience = 2  # Number of epochs to wait for improvement
best_accuracy = 0
epochs_without_improvement = 0
epoch = 0

while epochs_without_improvement < patience and epoch < max_epochs:
    print('epoch', epoch)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (spectrograms, labels) in enumerate(train_loader):
        print(f'\rProcessing batch {i+1}/{len(train_loader)}', end='', flush=True)
        
        # Clear memory cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         
        spectrograms, labels = spectrograms.to(device, non_blocking=True), labels.to(device, non_blocking=True) # non_blocking=True is allows the data to be loaded in parallel
        
        optimizer.zero_grad(set_to_none=True) # uses None instead of 0 to reduce memory usage
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
                
        # Explicitly clear some variables
        del outputs
        del loss
    
    print('')  # New line after batch processing is complete
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Check if accuracy improved
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        
    print(f'Best accuracy so far: {best_accuracy:.2f}%')
    print(f'Epochs without improvement: {epochs_without_improvement}')

    # Create models directory if it doesn't exist
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model checkpoint
    checkpoint_path = f'models/model_checkpoint_epoch_{epoch+1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy
    }, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')
    
    epoch += 1

print(f'\nTraining stopped after {epoch} epochs')
if epochs_without_improvement >= patience:
    print(f'Early stopping triggered: no improvement for {patience} consecutive epochs')
elif epoch >= max_epochs:
    print('Maximum number of epochs reached')