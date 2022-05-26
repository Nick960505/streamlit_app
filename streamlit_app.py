import time
import streamlit as st
import numpy as np
import pandas as pd
import librosa
from python_speech_features import mfcc
import tensorflow as tf
import numpy as np
from os import listdir

def get_class_lb(y):
    Y=[]
    for i in range(len(y)):
        x=np.where(y[i] == np.amax(y[i],axis=0))
        Y.append(x[0][0])
    return Y


st.title('This is my application for KWS and SV')

st.write("I am Lawrence")

sr_number = 16000
offset=5000
data_size=70000


model_path = "./../Voice_work_14p/model.tflite"
SOUND_DIR = "./../playlist/"
f=listdir(SOUND_DIR)
st.write(f)
st.write('Now read :',f[1])
data, samplerate=librosa.load('./../playlist/yes.wav', sr=sr_number)



featrue_data = mfcc(data, samplerate)
inference_data = np.zeros((1,99,13,1))
inference_data[0:featrue_data.shape[0]]=np.array(featrue_data).reshape(99,13,1)

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

class_names={0:'yes',1:'no',2:'on',3:'off',4:'yes'}

cls=class_names[PD[0]]


st.write("I think you said :",cls)