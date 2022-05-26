import queue
import time
import numpy as np
import pandas as pd
import librosa
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


os.environ['NUMEXPR_MAX_THREADS'] = '16'

def get_class_lb(y):
    Y=[]
    for i in range(len(y)):
        x=np.where(y[i] == np.amax(y[i],axis=0))
        Y.append(x[0][0])
    return Y

image = Image.open('./Logo.png')

st.image(image, caption='Fitipower')


st.title('This is the application for KWS and SV')

#st.write("I am Lawrence")

Mode=st.radio("Choose the Mode",["Enroll Mode","User Mode"])

if Mode=="Enroll Mode":
    name = st.text_input("Your name : ")
    st.write(name)
else:
    pass


#################### model  & data
sr_number = 16000
offset=5000
data_size=70000

model_path = "./../app_model/model.tflite"
SOUND_DIR = "./../playlist/"
f=listdir(SOUND_DIR)
st.write(f)

data_upload=st.file_uploader("Upload a wav file",type=['wav'])

if data_upload is not None:
    #st.write('Now read :temp.wav')
    st.write("Now read :"+data_upload.name)
    data, samplerate=librosa.load('./../playlist/temp.wav', sr=sr_number)

    data=data[:16160]

    featrue_data = mfcc(data, samplerate)
    inference_data = np.zeros((1,100,13,1))
    inference_data[0:featrue_data.shape[0]]=np.array(featrue_data).reshape(100,13,1)

    m=-119.723
    M=110.061
    inference_data = (inference_data-m)/(M-m)
    inference_data=np.clip(inference_data,0,1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    print()
    print("Input details:")
    print(input_details)
    print()
    print("Output details:")
    print(output_details)
    print()
    input_type = input_details[0]['dtype']

    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        print("Input scale:", input_scale)
        print("Input zero point:", input_zero_point)
        print()
        inference_data = (inference_data / input_scale) + input_zero_point
        inference_data = np.around(inference_data)
        
    inference_data = inference_data.astype(input_type)
    #inference_data = np.expand_dims(inference_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], inference_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    output_type = output_details[0]['dtype']
    if output_type == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        print("Raw output scores:", output)
        print("Output scale:", output_scale)
        print("Output zero point:", output_zero_point)
        print()
        output = output_scale * (output.astype(np.float32) - output_zero_point)

    O=output.astype(np.float32)

    PD=get_class_lb(O)

    cls_name_list=['one','two','three','four','five','six','seven','eight','nine','zero','up','down','left','right','on','off','stop','go','yes','no']
    cls_name_list.sort()

    class_names={}
    for c in range(len(cls_name_list)):
        class_names[c]=cls_name_list[c]

    st.write(class_names)

    cls=class_names[PD[0]]


    st.write("I think you said :",cls)

data_upload=None
################################################################################



def main():
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
            status_indicator.write("AudioReciver is not set. Abort. Wait please!")

            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:

        st.info("Writing wav to disk???")
        
        
        saved=st.button("SAVE THIS!")
        remove=st.button("GIVE UP!")
        
        while True:
            if saved:
                file_name = st.text_input("file name : ")
                audio_buffer.export("./../playlist/"+file_name+".wav", format="wav")

                with open("./../playlist/temp.wav", 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)
            if remove:
                break
        
        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()


if __name__ == "__main__":
    main()