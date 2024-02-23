import numpy as np 
import pandas as pd 
import random
import os
from tqdm import tqdm
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import librosa.effects as le
import IPython.display as ipd
from tensorflow import reshape
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from itertools import cycle
import joblib
from flask import Flask,render_template,request


UPLOAD_FOLDER = 'uploads'


app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


m1=joblib.load('static/mods/Emotion_Audio_Model.pkl')
m2=joblib.load('static/mods/Emotion_Audio_Model2.pkl')
m3=joblib.load('static/mods/Emotion_Audio_Model3.pkl')
m4=joblib.load('static/mods/Emotion_Audio_Model4.pkl')
m5=joblib.load('static/mods/Emotion_Audio_Model5.pkl')


def predict_model(audio1):       
    target_shape=(128,128)
    y, sr =librosa.load(audio1, sr=None)
    y_stretched = le.time_stretch(y, rate=1)
    Mel_spectrogram = librosa.feature.melspectrogram(y=y_stretched , sr=sr)
    Mel_spectrogram = resize(np.expand_dims(Mel_spectrogram,axis=-1),target_shape)
    Mel_spectrogram = reshape(Mel_spectrogram, (1,) + target_shape + (1,))
    # print('it predicted model')
    return Mel_spectrogram

def models_system(MSG):
    predictions = m1.predict(MSG)
    class_probabilities = predictions[0]
    predicted_class_index = np.argmax(class_probabilities)

    if predicted_class_index==0:#negative
        predictions = m2.predict(MSG)
        class_probabilities = predictions[0]
        predicted_class_index = np.argmax(class_probabilities)

        if predicted_class_index==0:#model5
            predictions = m5.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Angry"
            elif predicted_class_index==1:
                Final_pred="Happy"   
            
        elif predicted_class_index==1:
            Final_pred="Disgusted"
        elif predicted_class_index==2:
            Final_pred="Fearful"        
        elif predicted_class_index==3:#model4
            predictions = m4.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Sad"
            elif predicted_class_index==1:
                Final_pred="Neutral"

    elif predicted_class_index==1:#positive
        predictions = m3.predict(MSG)
        class_probabilities = predictions[0]
        predicted_class_index = np.argmax(class_probabilities)

        if predicted_class_index==0:#model5
            predictions = m5.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Angry"
            elif predicted_class_index==1:
                Final_pred="Happy"     
            
            
        elif predicted_class_index==1:#model4
            predictions = m4.predict(MSG)
            class_probabilities = predictions[0]
            predicted_class_index = np.argmax(class_probabilities)
            if predicted_class_index==0:
                Final_pred="Sad"
            elif predicted_class_index==1:
                Final_pred="Neutral"
    # print('it gave the aswer')
    return Final_pred


@app.route('/')
def main():
	return render_template('index.html')

@app.route('/result',methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        audio_file = request.files['audioFile']
        if audio_file:
            save_path = 'uploads/' + audio_file.filename
            audio_file.save(save_path)
            print('Audio file uploaded successfully.')
            resdata=predict_model(save_path)
            result=models_system(resdata)
            return render_template('result.html',result=result)
        else:
            return 'No audio file provided.', 400
# main driver function
if __name__ == '__main__':
	# run() method of Flask class runs the application 
	# on the local development server.
	app.run()