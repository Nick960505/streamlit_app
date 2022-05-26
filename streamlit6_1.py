
import streamlit as st
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
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
import queue
import time


@st.cache
def infer(Model="./../app_model/model.tflite",inference_data=0):
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


@st.cache
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
            media_stream_constraints={
                "audio": True,
            },
        ),
    )
    

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()
    KWS_status_indicator = st.empty()
   

    while True:

        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something! ✔️ ✔️ ✔️")
            sound_chunk = pydub.AudioSegment.empty()
            KWS_status_indicator.empty()

            for audio_frame in audio_frames:
                KWS_status_indicator.header(round(time.time()-time_0,3))
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
            status_indicator.write("AudioReciver is not set. 🔇")
            break
    audio_buffer = st.session_state["audio_buffer"]

    time_0=time.time()
    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        st.write("len(audio_buffer)",len(audio_buffer))
        st.info("Writing wav to %s folder" %save_path)
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M")
        audio_buffer.export("./%s/temp_%s.wav" %(save_path,dt_string), format="wav")
        
        with open("./%s/temp_%s.wav" %(save_path,dt_string), 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.write(len(audio_bytes))
            st.header("temp_%s.wav" %dt_string)
            st.audio(audio_bytes)
        
        data, samplerate = librosa.load("./%s/temp_%s.wav" %(save_path,dt_string), sr=16000)
        #plt.plot(data)
        #st.pyplot()
        fig,ax=plt.subplots()
        ax.plot(data)
        st.pyplot(fig)

        ########################################## Analysis
        waveform=np.abs(data)
        waveform.sort()
        v=waveform[-100:-99]

        if v > 0.01 :
            origin_data_size=len(data)
            st.write("origin_data_size",origin_data_size)
            #st.write(origin_data_size>20000)
            offset=np.where(data>0.04)[0][0]
            st.write("offset",offset)
            data=data[offset-1160:offset+15000]
            if (offset+15000) < origin_data_size:
                featrue_data = mfcc(data, samplerate)
                inference_data = np.zeros((1,100,13,1))
                inference_data[0:featrue_data.shape[0]]=np.array(featrue_data).reshape(100,13,1)

                m=-119.723
                M=110.061

                inference_data = (inference_data-m)/(M-m)
                inference_data=np.clip(inference_data,0,1)
                O = infer(Model="./../app_model/model_21KW.tflite",inference_data=inference_data)
                PD=get_class_lb(O)
                cls=class_names[PD[0]]
                result = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">The Key word is : %s</p>' %cls
                st.markdown(result,unsafe_allow_html=True)

            else:
                status_indicator.header("Do not catch enough utterence")
                KWS_status_indicator.empty()
        else :
 
                    status_indicator.header("Please say loudly")
                    KWS_status_indicator.empty()
                    ###
                    #flag_0 = time.time()

            
                #time_4 = max(2 - flag_0,0)
                time.sleep(2)
                status_indicator.header('Ready')
               
                #KWS_status_indicator.empty()
                for i in range(3):
                    KWS_status_indicator.header('%s' %(3-i))
                    time.sleep(1)
                


                os.remove("temp.wav")
                st.session_state["audio_buffer"] = pydub.AudioSegment.empty()
                st.session_state.check=True


        else:
            status_indicator.write("AudioReciver is not set. 🔇")
            break

    if not webrtc_ctx.state.playing:
        pass



    


cls_name_list=['one','two','three','four','five','six','seven','eight','nine','zero','up','down','left','right','on','off','stop','go','yes','no','_background_noise_']
cls_name_list.sort()
class_names={}
for c in range(len(cls_name_list)):
    class_names[c]=cls_name_list[c]



image = Image.open('./Logo.png')

st.image(image, caption='Fitipower',width=500)

st.title('This is the application for Key Word Spotting and Speaker Verification')
st.write("\n\n\n")
next = st.sidebar.button("Next option")

st.sidebar.markdown("---")

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
    st.header("Now is Enroll Mode :smile:")
    st.markdown("---")

    col1,col2 = st.columns([1,1])

    with col1:
        st.write("\n\n")
        Init = st.button("Initiate the Member list",key="init")
        if Init:
            df=pd.read_csv('test.csv')#,index_col="ID"
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            #new_df=df.drop(df.columns, axis=1)
            #new_df=pd.DataFrame(df.index, columns=df.columns)
            #new_df.to_csv('test.csv')
            df.loc[[0]].to_csv('test.csv')
            st.write(df.loc[[0]])

    with col2:
        Name=st.text_input("Input your name: 👇")
    st.markdown("---")

    with st.expander("See explanation"):
        st.write('''
            Here you can record your voice and be the member
        ''')

    if len(Name)>=1:

        image = Image.open('./music.jpg')
        st.image(image, caption='record',width=500)

        if not os.path.exists("./enrollment/%s" %Name):
            os.mkdir("./enrollment/%s" %Name)
        st.write("Your name is ",Name," , ready to enroll")

        df=pd.read_csv('test.csv',index_col="ID")#.iloc[:,:]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        st.write(df)

        if Name in set(df.Name):
            st.write("Already Member or This name has been used")
        else:
            ID=len(df)
            st.write("You are the No.%s Member here, welcome! " %(ID))
            # Default
            
            now = datetime.now()
            #now =  now.strftime("%Y_%m_%d_%H_%M")
            Date = now.strftime("%Y_%m_%d")
            Time = now.strftime("%H_%M")
            Mean_Vec=[0]*10
            member_info=[]
            member_info.append([[Name],[Date],[Time],Mean_Vec])
            member_info = [item for sublist in member_info for item in sublist]
            member_info = [item for sublist in member_info for item in sublist]


            df.loc[ID]=member_info
            #df.columns=['Name']
            df=df.astype('str')
            st.write(df)
            #df.to_csv('test.csv')
            st.session_state.check=False
            main(save_path="enrollment/%s" %Name)

            if st.session_state.check==True:
                df.to_csv('test.csv')
                
            #


            ############################ 
    


elif Mode=='User Mode':

    Task=st.radio("Choose the Task",["Key Word Spotting","Speaker Verification"],key="radio_option_user",index=0)
    st.markdown("---")
    if Task == "Key Word Spotting":
        st.header("Now is User Mode of %s :smile:" %Task)
        option = st.selectbox('Which way? 👇 ',('Record', 'Upload'),index=1)
        st.markdown("---")


        if option == "Record":
            #st.subheader("It takes about 1 minute. Please wait awhile. 🙏🏽")
            #start_button=st.button("START:")
            #if start_button:
            main()
                


        if option == "Upload":
            audio_folder = "playlist"
            filenames = glob.glob(os.path.join(audio_folder, "*.wav"))
            selected_filename = st.selectbox("Select a file 👇 ", filenames)
            st.markdown("---")
            go_button = st.button('GO!')
            
            if selected_filename is not None:
                if go_button:
                    in_fpath = Path(selected_filename.replace('"', "").replace("'", ""))
                    data, samplerate = librosa.load(str(in_fpath), sr=16000)
                    origin_data_size=len(data)
                    data=data[:16160]
                    featrue_data = mfcc(data, samplerate)
                    fig,ax=plt.subplots()
                    ax.plot(data)
                    st.pyplot(fig)
                    

                    if featrue_data.shape[0]<95:
                        st.write("featrue_data.shape",featrue_data.shape)
                        st.write("The utterance is too short! Choose the other file please! 🙏🏽")

                    elif origin_data_size<48000:

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
                        O = infer(Model="./../app_model/model_21KW.tflite",inference_data=inference_data)
                        PD=get_class_lb(O)
                        #PD
                        #st.write(class_names)
                        #PD[0]
                        cls=class_names[PD[0]]
                        result = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">The Key word is : %s</p>' %cls
                        st.markdown(result,unsafe_allow_html=True)
                        #st.markdown('The Key word is : **%s**'  %cls)

                    else :
                        #st.write("featrue_data.shape",featrue_data.shape)
                        st.write("The utterance is too long! Choose the other file please! 🙏")


    else:
        st.write("Speaker Verification")


    with st.expander("See explanation for Key Word Spotting 👋🏼"):
            st.write('''
                This Mode has two ways to check the keywords, \n
                first one is to upload the current wav files directly, \n
                the other one is to record your voice in real time \n
                but it may take you longer time! \n
                After you confirm there is no problem with the record, \n
                upload the file to check the keywords. \n
                Now. Let's start! 👌🏽 \n
                Please press start when you ready to record :smile:
            ''')







else:
    df=pd.read_csv('test.csv',index_col="ID").astype('str')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    st.write(df)
    st.balloons()
    image = Image.open('./wave.jpg')
    st.image(image, caption='📃▶️',width=500)

    
