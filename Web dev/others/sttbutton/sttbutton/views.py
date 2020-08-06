from django.shortcuts import render
import requests


def button(request):

    return render(request,'home.html')

def output(request):
    import pyaudio
    import wave
    import deepspeech
    import numpy as np
    import os
    import time

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    DEEPSPEECH_MODEL_DIR ="/Users/xinyunwu/Desktop/stt/stt"
    MODEL_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR,"deepspeech-0.7.4-models.pbmm")
    model = deepspeech.Model(MODEL_FILE_PATH)
    filename="output.wav"
    w = wave.open(filename, 'r')
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    text = model.stt(data16)

    
    
    data=text
    return render(request,'home.html',{'data':data})

