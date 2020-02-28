# Script that automatically extacts audio features from mp3s contained in the "data folder"
# Saves a .csv file with features to disk

import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import pandas as pd

# Computes a bunch of audio features and returns a dictionary
def AudioFeatures(y,sr,class_label):
    # Dictionary in which to store features
    feature_dict = {}
    
    # Compute means of features
    feature_dict['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    feature_dict['spec_cent'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    feature_dict['spec_bw'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    feature_dict['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    feature_dict['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
     
    # Computes MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = [np.mean(x) for x in mfcc]

    for i in range(0,20):
        feature_dict['mfcc'+str(i)] = mfcc_mean[i]
        
    # Class label
    feature_dict['class'] = class_label
    
    return feature_dict

# Splits up audio file into chunks of some length, and extract audio features from those
def WindowExtract(y,sr,class_label,window_length_sec = 10):
    # List to store dictionaries of audio features
    audio_features_list = []

    # Divides up the audio file into chunks
    audio_len_sec = len(y)/sr # Total length of audio in seconds
    n_windows = int(audio_len_sec // window_length_sec) # Number of windows

    for i in range(0,n_windows+1):
        # Defines window beginning/end
        window_start = sr*i*window_length_sec
        window_end = sr*(i+1)*window_length_sec

        # Defines window
        window = y[window_start:window_end]

        feature_dict = AudioFeatures(window,sr,class_label)
        audio_features_list.append(feature_dict)
        
    return audio_features_list

# Given a dictionary of filenames and classes, process all the audio files contained therein 
def ProcessAudioFiles(mp3_classes_input,window_length_sec = 10):
    all_audio_features = []
    
    # Loops over all files
    for filepath,class_label in mp3_classes_input.items():
        print('Processing:',filepath)
        y , sr = librosa.load(filepath,mono=True)
        audio_features_list = WindowExtract(y,sr,class_label,window_length_sec)
        all_audio_features = all_audio_features+ audio_features_list
        
    print('Done processing')
    
    return all_audio_features

# Fetches file paths from folder
import glob
import re

training_file_paths = glob.glob("data/*/*.mp3") # .mp3 files inside folder
folder_paths = glob.glob("data/*/") # Names of folders

# Creates dictionary with mp3 filepaths and class
mp3_classes = {}
for file in training_file_paths:
    mp3_classes[file] = re.search(r'\/.*?\/',file).group(0).replace('/','')   

print('Files to be processed:')
for x in mp3_classes.keys():
	print(x)

# Processes all audiofiles in dictionary
all_audio_features = ProcessAudioFiles(mp3_classes)

# Creates a dataframe from list of dictionaries
audio_features_df = pd.DataFrame(all_audio_features)

# Save dataframe as csv
audio_features_df.to_csv('audio_features.csv',index = False)