
import streamlit as st
from pathlib import Path
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import os
import librosa
import glob
import pydub
import pandas as pd
from PIL import Image
from datetime import datetime
import tensorflow as tf

def infer(Model="./app_model/model.tflite",inference_data=0):
    interpreter = tf.lite.Interpreter(model_path=Model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    input_type = input_details[0]['dtype']

    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        inference_data = (inference_data / input_scale) + input_zero_point
        inference_data = np.around(inference_data)

    inference_data = inference_data.astype(input_type)
    interpreter.set_tensor(input_details[0]['index'], inference_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    output_type = output_details[0]['dtype']

    if output_type == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output = output_scale * (output.astype(np.float32) - output_zero_point)
    O=output.astype(np.float32)
    return O

def get_class_lb(y):
    Y=[]
    for i in range(len(y)):
        x=np.where(y[i] == np.amax(y[i],axis=0))
        Y.append(x[0][0])
    return Y

def main(save_path="playlist"):
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "audio": True,
            },
        ),
    )

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                st.session_state["audio_buffer"] += sound_chunk
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        st.info("Writing wav to %s folder" %save_path)
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M")
        audio_buffer.export("./%s/temp_%s.wav" %(save_path,dt_string), format="wav")
        
        with open("./%s/temp_%s.wav" %(save_path,dt_string), 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.write(len(audio_bytes))
            st.audio(audio_bytes)
        
        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
        st.session_state.check=True

    





cls_name_list=['one','two','three','four','five','six','seven','eight','nine','zero','up','down','left','right','on','off','stop','go','yes','no']
cls_name_list.sort()
class_names={}
for c in range(len(cls_name_list)):
    class_names[c]=cls_name_list[c]



image = Image.open('./Logo.png')

st.image(image, caption='Fitipower',width=500)

st.title('This is the application for KWS and SV')

next = st.sidebar.button("Next option")

if next:
    if st.session_state["radio_option"] == "Enroll Mode":
        st.session_state.radio_option = "User Mode"
    elif st.session_state["radio_option"] == "User Mode":
        st.session_state.radio_option = "DataBase"
    else:
        st.session_state.radio_option = "Enroll Mode"


Mode=st.sidebar.radio("Choose the Mode",["Enroll Mode","User Mode","DataBase"],key="radio_option",index=1)


#'session_state',st.session_state

if Mode=='Enroll Mode':
    st.write("Now is Enroll Mode :smile:")
    with st.expander("See explanation"):
        st.write('''
            Here you can record your voice and be the member
        ''')
    Name=st.text_input("Input your name:")
    
    if len(Name)>=1:
        if not os.path.exists("./enrollment/%s" %Name):
            os.mkdir("./enrollment/%s" %Name)
        st.write("Your name is ",Name," , ready to enroll")

        df=pd.read_csv('test.csv',index_col="ID")#.iloc[:,:]
        st.write(df)

        if Name in set(df.Name):
            st.write("Already Member or This name has been used")
        else:
            ID=len(df)
            st.write("You are the No.%s Member here, welcome! " %(ID))
            df.loc[ID]=[Name]
            df.columns=['Name']
            st.write(df)
            #df.to_csv('test.csv')
            st.session_state.check=False
            main(save_path="enrollment/%s" %Name)

            if st.session_state.check==True:
                df.to_csv('test.csv')
                
            #


            ############################ 
    


elif Mode=='User Mode':
    st.write("Now is User Mode :smile:")
    with st.expander("See explanation"):
        st.write('''
            This Mode has two ways to check the keywords,
            first one is to upload the current wav files,
            the other one is to record your voice in real time
            but it may take you longer time! \n
            Please press start when you ready to record
        ''')
    option = st.selectbox('Which way?',('Record', 'Upload'))

    if option == "Record":
        #start_button=st.button("START:")
        #if start_button:
        main()











    if option == "Upload":
        
        #go_button = st.button('GO!')
        '''
        data_upload=st.file_uploader("Upload a wav file in playlist",type=['wav'])
        if data_upload is not None:
            if go_button:
                st.write("Now read :"+data_upload.name)
                data, samplerate=librosa.load('./playlist/'+data_upload.name, sr=16000)
                data=data[:16160]
                featrue_data = mfcc(data, samplerate)
                featrue_data
                inference_data = np.zeros((1,100,13,1))
                inference_data[0:featrue_data.shape[0]]=np.array(featrue_data).reshape(100,13,1)
                m=-119.723
                M=110.061
                inference_data = (inference_data-m)/(M-m)
                inference_data=np.clip(inference_data,0,1)
        '''
        audio_folder = "playlist"
        filenames = glob.glob(os.path.join(audio_folder, "*.wav"))
        selected_filename = st.selectbox("Select a file", filenames)
        go_button = st.button('GO!')
        
        if selected_filename is not None:
            if go_button:
                in_fpath = Path(selected_filename.replace('"', "").replace("'", ""))
                data, samplerate = librosa.load(str(in_fpath), sr=16000)
                data=data[:16160]
                featrue_data = mfcc(data, samplerate)
                if featrue_data.shape[0]<95:
                    st.write("featrue_data.shape",featrue_data.shape)
                    st.write("The utterance is too short! Choose the other file please!")
                else:
                    "(Before)featrue_data.shape",featrue_data.shape
                    featrue_data=np.lib.pad(featrue_data, ((0,100-featrue_data.shape[0]),(0,0)), 'constant', constant_values=(0))
                    "(After)featrue_data.shape",featrue_data.shape
                    inference_data = np.zeros((1,100,13,1))
                    inference_data[0:featrue_data.shape[0]]=np.array(featrue_data).reshape(100,13,1)

                    m=-119.723
                    M=110.061
                    
                    inference_data = (inference_data-m)/(M-m)
                    inference_data=np.clip(inference_data,0,1) 
                    st.write("inference_data.shape",inference_data.shape)
                    # display the st.audio of inference 
                    #'before infer',inference_data  
                    #'after infer' ,infer(0,inference_data) 
                    O = infer(Model="./../app_model/model.tflite",inference_data=inference_data)
                    PD=get_class_lb(O)
                    #PD
                    #st.write(class_names)
                    #PD[0]
                    cls=class_names[PD[0]]
                    result = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">The Key word is : %s</p>' %cls
                    st.markdown(result,unsafe_allow_html=True)
                    #st.markdown('The Key word is : **%s**'  %cls)







else:
    df=pd.read_csv('test.csv',index_col="ID")
    st.write(df)

    
