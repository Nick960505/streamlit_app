import queue
import time
import pandas as pd
from python_speech_features import mfcc
import tensorflow as tf
from os import listdir
import pydub
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from IPython.display import clear_output
from IPython import display
from PIL import Image
import os
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import pathlib
import librosa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import librosa.display
from python_speech_features.base import logfbank,mfcc
import warnings
from matplotlib import colors as mcolors
from sklearn import metrics
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date
import datetime
from scipy.spatial import distance as ds
import json
import SessionState
import python_2



image = Image.open('./Logo.png')
st.image(image, caption='Fitipower')
st.title('This is the application for KWS')

Mode=st.sidebar.radio("Choose the Mode",["Rest","Enroll Mode","User Mode"])

while True:

    if Mode=="Enroll Mode":
        st.write("Now is Enroll Mode")
        read='yes'

        while True:
                
            if read!='yes':
                break

            database="enrollment.json"
            wav_file='enroll.wav'
            init = st.text_input("init if initiate:")

            if init=="init":
                ### initiate
                User={}
                Erollment={}
                open(database, 'w').close()

            else:
                with open(database) as f:
                    User={}
                    #Erollment = json.load(f.text)
                    #Erollment = json.load(f)
                open(database, 'w').close()

            Enroll_name=st.text_input("Enroll_name's input:")
            today=datetime.datetime.today()
            time=(today.year,today.month,today.day,today.hour) ## latest update
            count=6
            st.write(Enroll_name)

            while read=='yes':
                    
                    if os.path.exists(wav_file):
            
                        DV_mean=np.load('output_layer_300_test_2.npy')[0].astype(np.float)[:10]


                        User['time']=time
                        User['count']=count
                        User['DV_mean']=list(DV_mean)

                        Erollment['%s' %Enroll_name]=User

                        with open(database, "a+") as f:
                            json.dump(Erollment, f, indent = 4)

                        st.write("Saved! Number of members : ",len(Erollment))

                        closed = st.text_input("enrollment closed or not:")

                        if os.path.exists(wav_file):
                            ###
                            st.write("remove last wav file")
                            #os.remove(wav_file)
                            ###

                        ### if stop
                        if closed=='yes':
                            break
    elif Mode=="User Mode":

        st.write("Now is User Mode")
        user_file="user.wav"

        closed = st.text_input("device closed or not:")

        if closed=='yes':
            break
    else:
        break