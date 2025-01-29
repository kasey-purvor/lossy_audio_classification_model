import zipfile
from lossy_audio_classification_model.src.data_handling.helper_functions import process_and_save_track, is_suitable_file, is_training_track

train_data_path = 'data/raw/'
processed_data_path = 'data/processed_data/'

def extract_musdb_mixture_files(raw_data_path, processed_data_path):
    """Extract and process audio files from MUSDB dataset."""
    counter = 0
    print('extracting files from zip', raw_data_path + 'musdb18hq.zip')
    with zipfile.ZipFile(raw_data_path + 'musdb18hq.zip', 'r') as zip_ref:
        for track in zip_ref.namelist():
            components = track.split('/')
                
            if is_suitable_file(components):
                is_training = is_training_track(components)
                track_name = components[1]
                print('Is training = ', is_training, '|  trackname = ', track_name)
                
                process_and_save_track(zip_ref, track, track_name, processed_data_path, is_training)
            
                counter += 1
                print(counter, 'files processed')
                print('----------------------------------------')
    print('All audio files processed. Saving to disk')
    
extract_musdb_mixture_files(train_data_path, processed_data_path)

