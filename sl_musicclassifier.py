import streamlit as sl
import librosa

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

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

def WindowExtract(y,sr,class_label,window_length_sec = 10):
    # List to store dictionaries of audio features
    audio_features_list = []

    # Divides up the audio file into chunks with a rolling window
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

# 

sl.title('Audio-based musical instrument detection')

filepath = sl.text_input('Choose an audio file (filepath):',value='data/rach_prelude.mp3')


if sl.button('Predict!'):
	# Loads file
	sl.write('Loading file...')
	y_violin , sr = librosa.load(filepath,mono=True)
	sl.write('Analyzing audio features...')

	# Extracts audio features
	audio_features = AudioFeatures(y_violin,sr,'unknown')

	# Creates a dataframe of features
	audio_features_df = pd.DataFrame([audio_features])

	# Load classifier model from pickle file
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.preprocessing import LabelEncoder
	from sklearn import decomposition

	rf,pca, Encoder = pickle.load(open('model.pkl', 'rb'))

	# Does PCA transformation
	X = audio_features_df.iloc[:,:-1]
	X = pca.transform(X)

	# Makes prediction
	predictions = rf.predict(X)
	predicted_label = Encoder.inverse_transform(predictions)
	sl.header('Prediction: '+str(predicted_label[0]))
	sl.subheader('PCA Visualization')
	# 2D PCA visualization
	pca_2d = pickle.load(open('PCA_2D.pkl', 'rb'))

	trained_audio_features_df = pd.read_csv('audio_features.csv')
	classes = trained_audio_features_df['class'].unique()

	# Plots points for every class
	for x in classes:
	    pca_result = pca_2d.transform(trained_audio_features_df[trained_audio_features_df['class']==x].iloc[:,:-1].values)      
	    plt.scatter(pca_result[:,0],pca_result[:,1],s=10,label=x)

	# Plots predicted location on PCA plot
	predicted_pca_result = pca_2d.transform(audio_features_df.iloc[:,:-1].values)
	plt.scatter(predicted_pca_result[:,0],predicted_pca_result[:,1],s=100,marker='x',label='prediction')

	plt.legend(loc=0)
	plt.xlim([-2000,2500])
	plt.ylim([-300,400])
	plt.grid(True)
	sl.pyplot()