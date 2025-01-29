# Create a FastAPI endpoint for model inference
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import io
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.neural_networks.cnn import AudioClassifier
from data_handling.dataset_classes import InferenceDataset
from data_handling.helper_functions import create_lossless_waveform, generate_spectrograms, split_waveform

app = FastAPI()

# Load the classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier()
model.to(device)
model.eval()

# Load the latest checkpoint
checkpoint_files = [f for f in os.listdir('models/checkpoints') if f.startswith('model_checkpoint_epoch_')]
if checkpoint_files:
    
    # Get the latest checkpoint file
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join('models/checkpoints', latest_checkpoint)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']

    print(f'Loaded checkpoint from {checkpoint_path}')
    print(f'Epoch: {start_epoch}, Loss: {checkpoint["loss"]:.4f}, Accuracy: {checkpoint["accuracy"]:.2f}%')
else:
    print('No checkpoint found. Starting from scratch.')

@app.post("/predict")
async def predict(request: Request):
    try:
        # Read raw binary data from request body
        wav_data = await request.body()
        wav_buffer = io.BytesIO(wav_data)
        
        # Create lossless waveform from WAV file
        waveform, sr = create_lossless_waveform(wav_buffer)
        
        # split waveform into segments
        segments = split_waveform(waveform, sr)
        
        # Generate spectrogram
        spectrograms = generate_spectrograms(segments)

        # Create inference dataset and dataloader
        inference_dataset = InferenceDataset(spectrograms)
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=16,
            shuffle=False,
            pin_memory=True
        )
        
        # Get predictions for all segments
        predictions = []
        confidences = []
        for batch in inference_loader:
            batch = batch.to(device)
            with torch.no_grad():
                output = model(batch)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                predictions.extend(predicted.cpu().numpy())
                confidences.extend(probs.cpu().numpy())
        
        # Aggregate predictions
        predictions = np.array(predictions)
        confidences = np.array(confidences)

        prediction_string = f"{np.sum(predictions == 0)} / {len(predictions)} predictions are 'Lossless'."
        
        # Take majority vote
        final_prediction = np.bincount(predictions).argmax().item()
        if final_prediction == 0:
            final_prediction = 'Lossless'
        else:
            final_prediction = 'Lossy'
        
        # Get average confidence for predicted class
        final_confidence = float(np.mean([conf[pred] for conf, pred in zip(confidences, predictions)]))
        
        
        
        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            'predictions': prediction_string,
            # 'confidences': confidences.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(
        'app:app',
        host="0.0.0.0",
        port=8000,
        reload=True
    )
