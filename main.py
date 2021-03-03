import time
import wave
import sys
import os

import numpy as np
from pydub import AudioSegment
from pydub.playback import play

import pyaudio
# Import models: here we add new AI models.
from aimodels.speechenhancement.speechenhancement import load_model, prediction

"""
GLOBAL CONFIGS.
"""
print("[GLOBAL CONFIGS] Loading....")
# channels = 1  # Mono
sample_rate = 44100  # Sample read to read and write
# periodsize = 320
# FORMAT = alsaaudio.PCM_FORMAT_S16_LE
# COUNTER_CACHE_LOOPS = 1000
LOAD_WEIGHTS = 0  # Not weights loaded
loaded_model = 0  # Loaded module, an object not a number
chunk = 1024  # set the chunk size of 1024 samples
FORMAT_INPUT = pyaudio.paInt32  # default LE_32bits or pyaudio.paInt32
FORMAT_OUTPUT = pyaudio.paFloat32  # default LE_32bits or pyaudio.paFloat32
channels = 1  # default mono
sample_rate_input = 44100  # default 44100
sample_rate_output = 8000  # default 8000 due restrictions, can't be more by 2021-03-02
record_seconds = 1.1  # default 1.1 due some restrictions
print("[GLOBAL CONFIGS] Completed.")

"""
FUNCTIONS
"""
def record_one_second(filename_path='temporal/input.wav'):
    """
    This script record around one second with pyaudio with specific finename_path
    Example taken from https://www.thepythoncode.com/article/play-and-record-audio-sound-in-python
    """
    p = pyaudio.PyAudio()  # initialize PyAudio object
    stream = p.open(format=FORMAT_INPUT,
                    channels=channels,
                    rate=sample_rate_input,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)  # open stream object as input & output
    frames = []
    print("[RECORD] Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("[RECORD] Finished recording.")

    stream.stop_stream()  # stop and close stream
    stream.close()
    p.terminate()  # terminate pyaudio object
    
    wf = wave.open(filename_path, "wb") # save audio file in 'write bytes' mode
    wf.setnchannels(channels)  # set the channels
    wf.setsampwidth(p.get_sample_size(FORMAT_INPUT))  # set the sample format
    wf.setframerate(sample_rate)  # set the sample rate
    wf.writeframes(b"".join(frames))  # write the frames as bytes
    wf.close()  # close the file
    return True


def play_one_second(file='temporal/output.wav'):
    """
    This script play file
    Example adapted from https://people.csail.mit.edu/hubert/pyaudio/
    """
    print("[PLAY] Opening file")
    if (sys.platform=='windows' or 
    sys.platform=='macos'):
        sound = AudioSegment.from_file(file)
        play(sound)
    elif (sys.platform=='linux'):
        command_inference='python3 tools_example/pyalsaaudio_playback.py temporal/output.wav'
        result=os.popen(command_inference).read()


def executeprediction(aimodel):
    """
    This function is a manager of ai models applied
    like filters to noisy mic
    """
    global LOAD_WEIGHTS  # gets value from global variable
    global loaded_model  # gets value from global variable

    # Speech-Enhancement model
    if aimodel == 'speechenhancement':
        print("[AI MODEL] Speech-Enhancement detected:{}", aimodel)

        if LOAD_WEIGHTS == 0:  # Not weights loaded
            print("[AI MODEL] {} --- [LOAD MODELS] Loading weigths...".format(aimodel))
            loaded_model = load_model(weights_path='aimodels/speechenhancement/weights',
                                      name_model='model_unet',
                                      audio_dir_prediction='temporal',
                                      dir_save_prediction='temporal/',
                                      audio_input_prediction=['input.wav'],
                                      audio_output_prediction='output.wav',
                                      sample_rate=8000,  # default 8000
                                      min_duration=1.0,  # default 1.0 second
                                      frame_length=8064,  # default 8064
                                      hop_length_frame=8064,  # default 8064
                                      n_fft=255,  # default 255
                                      hop_length_fft=63)  # default 63
            LOAD_WEIGHTS = 1
        if LOAD_WEIGHTS == 1:  # LOAD_WEIGHTS ALREADY LOADED TO GLOBAL VAR
            print(
                "[AI MODEL] {} --- [INFERENCE] Inference net, cleaning audio".format(aimodel))
            prediction(weights_path='aimodels/speechenhancement/weights',
                       name_model='model_unet',
                       audio_dir_prediction='temporal',
                       dir_save_prediction='temporal/',
                       audio_input_prediction=['input.wav'],
                       audio_output_prediction='output.wav',
                       sample_rate=8000,  # default 8000
                       min_duration=1.0,  # default 1.0
                       frame_length=8064,  # default 8064
                       hop_length_frame=8064,  # default 8064
                       n_fft=255,  # default 255
                       hop_length_fft=63,  # default 63
                       loaded_model=loaded_model)  # pretrained model
            print("[AI MODEL] Speech-Enhancement finished.")
    # Here you can add a new models
    else:
        print("[AI MODEL] No aimodel selected")
        return False


def hello():
    print("""------------------------------------------------------------------------------------
 _____   ______   ______   __        ________   ______   __    __  ________  _______  
/      \ |      \ /      \ |  \      |        \ /      \ |  \  |  \|        \|       \ 
|  $$$$$$\ \$$$$$$|  $$$$$$\| $$      | $$$$$$$$|  $$$$$$\| $$\ | $$| $$$$$$$$| $$$$$$$\ 
| $$__| $$  | $$  | $$   \$$| $$      | $$__    | $$__| $$| $$$\| $$| $$__    | $$__| $$
| $$    $$  | $$  | $$      | $$      | $$  \   | $$    $$| $$$$\ $$| $$  \   | $$    $$
| $$$$$$$$  | $$  | $$   __ | $$      | $$$$$   | $$$$$$$$| $$\$$ $$| $$$$$   | $$$$$$$\ 
| $$  | $$ _| $$_ | $$__/  \| $$_____ | $$_____ | $$  | $$| $$ \$$$$| $$_____ | $$  | $$
| $$  | $$|   $$ \ \$$    $$| $$     \| $$     \| $$  | $$| $$  \$$$| $$     \| $$  | $$
\$$   \$$ \$$$$$$  \$$$$$$  \$$$$$$$$ \$$$$$$$$ \$$   \$$ \$$   \$$ \$$$$$$$$ \$$   \$$
                                                                                        
                                                                                        """)


if __name__ == '__main__':

    hello()

    while True:

        record_one_second()
        time.sleep(1)
        executeprediction('speechenhancement')
        time.sleep(1)
        play_one_second()
        time.sleep(1)
        # break
        #
