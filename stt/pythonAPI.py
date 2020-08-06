# A pythonAPI for using the DeepSpeech module : 
"""
	It works like a transcriber with two parts integrated together"
	- Record the user input continuously 
	- Streaming stt module 
"""

# Import :
import deepspeech
import numpy as np
import os

# If have problem downloading see the instruction.txt
import pyaudio

import time


# Deepspeech setups: might change with different model&input 
# In here i use the pre-trained english model and the default language model 

#DEEPSPEECH_MODEL_DIR = "/home/wxy/STT/"

DEEPSPEECH_MODEL_DIR ="/Users/xinyunwu/Desktop/stt/"
MODEL_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR,"deepspeech-0.7.4-models.pbmm")




#Deepspeech Module Object : using LM for better accuracy :
model = deepspeech.Model(MODEL_FILE_PATH)



# Streaming STT:

#1:  Open  a Streaming session
context = model.createStream()


#2: Feed Data: Callback function:
text_so_far = ''

def process_audio(in_data, frame_count, time_info, status):
    global text_so_far

    data16 = np.frombuffer(in_data, dtype=np.int16)
    #print(in_data)
    context.feedAudioContent(data16)
    text = context.intermediateDecode()

    '''
    if text != text_so_far:
        print('Interim text = {}'.format(text))
        text_so_far = text
    '''
    return (in_data, pyaudio.paContinue)



# PyAudio setups:
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1024


# PyAudio input streaming:
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
    stream_callback=process_audio
)



print('Please start speaking, when done press Ctrl-C ...')

stream.start_stream()



try: 

    while stream.is_active():

        time.sleep(0.1)

except KeyboardInterrupt:

    # Close straming"
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print('Finished recording.')

    # DeepSpeech
    text = context.finishStream()
    print('Final text = {}'.format(text))


# Todo : 
"""
    1: try to integrate with gui: like record buttom and done buttom 
    2: do as a function 
"""
