import getopt
import os
import struct
import sys
import time
import wave

import alsaaudio
import librosa
import numpy as np

from aimodels.speechenhancement.speechenhancement import load_model, prediction

"""
GLOBAL CONFIGS.
"""
print("[GLOBAL CONFIGS] Loading....")
device_input = 'sysdefault'
device_input_capture = alsaaudio.PCM_CAPTURE
device_input_type = alsaaudio.PCM_NONBLOCK
device_output = 'default'
device_output_capture = alsaaudio.PCM_PLAYBACK
channels = 1  # Mono
sample_rate = 44100  # Sample read to read and write
periodsize = 320
FORMAT = alsaaudio.PCM_FORMAT_S16_LE
COUNTER_CACHE_LOOPS = 1000
LOAD_WEIGHTS = 0  # Not weights loaded
loaded_model = 0  # Loaded module, an object not a number
audio_cache= bytearray() # Array of audio
print("[GLOBAL CONFIGS] Completed.")

"""
FUNCTIONS
"""


def sine_samples(freq, dur):
    # Get (sample rate * duration) samples on X axis (between freq
    # occilations of 2pi)
    X = (2*np.pi*freq/rate) * np.arange(rate*dur)

    # Get sine values for these X axis samples (as integers)
    S = (32767*np.sin(X)).astype(int)

    # Pack integers as signed "short" integers (-32767 to 32767)
    as_packed_bytes = (map(lambda v: struct.pack('h', v), S))
    return as_packed_bytes


def output_wave(path, frames):
    # Python 3.X allows the use of the with statement
    # with wave.open(path,'w') as output:
    #     # Set parameters for output WAV file
    #     output.setparams((2,2,rate,0,'NONE','not compressed'))
    #     output.writeframes(frames)

    output = wave.open(path, 'w')
    output.setparams((2, 2, sample_rate, 0, 'NONE', 'not compressed'))
    output.writeframes(frames)
    output.close()


def output_sound(path, freq, dur):
    # join the packed bytes into a single bytes frame
    frames = b''.join(sine_samples(freq, dur))

    # output frames to file
    output_wave(path, frames)

def h(data):
    """
    This function concatenate bytes into bytearrays
    """
    # Fasted array concatenation on python 
    # https://www.guyrutenberg.com/2020/04/04/fast-bytes-concatenation-in-python/
    
    return True

def executeprediction(aimodel):
    """
    This function is a manager of ai models applied
    like filters to noisy mic
    """
    global LOAD_WEIGHTS # gets value from global variable
    global loaded_model # gets value from global variable
    global audio_cache # gets value from global variable

    # Speech-Enhancement model
    if aimodel == 'speechenhancement':
        print("[AI MODEL] Speech-Enhancement detected")

        if LOAD_WEIGHTS:  # Not weights loaded
            print("[AI MODEL] {} --- Loading weigths...".format(aimodel))
            loaded_model = load_model(weights_path='aimodels/speechenhancement/weights',
                                      name_model='model_unet',
                                      audio_dir_prediction='temporal',
                                      dir_save_prediction='temporal/',
                                      audio_input_prediction=['input.wav'],
                                      audio_output_prediction='output.wav',
                                      sample_rate=8000, # default 8000
                                      min_duration=1.0,  # default 1.0 second
                                      frame_length=8064,  # default 8064
                                      hop_length_frame=8064,  # default 8064
                                      n_fft=255,  # default 255
                                      hop_length_fft=63)  # default 63
            LOAD_WEIGHTS = 1
        else:
            print("[AI MODEL] {} --- Inference net, cleaning audio".format(aimodel))
            prediction(weights_path='aimodels/speechenhancement/weights',
                       name_model='model_unet',
                       audio_dir_prediction='temporal',
                       dir_save_prediction='temporal/',
                       audio_input_prediction=['input.wav'],
                       audio_output_prediction='output.wav',
                       sample_rate=8000, # default 8000
                       min_duration=1.0,  # default 1.0
                       frame_length=8064,  # default 8064
                       hop_length_frame=8064,  # default 8064
                       n_fft=255,  # default 255
                       hop_length_fft=63,  # default 63
                       loaded_model=loaded_model)  # pretrained model

            return True

        print("[AI MODEL] Speech-Enhancement finished.")
    # Here you can add a new models
    else:
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
    # Open the device in nonblocking capture mode in mono,
    # with a sampling rate of 44100 Hz and 16 bit little endian samples
    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    # For our purposes, it is suficcient to know that reads from the device
    # will return this many frames. Each frame being 2 bytes long.
    # This means that the reads below will return either 320 bytes of data
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # mode.
    inp = alsaaudio.PCM(device_input_capture,
                        device_input_type,
                        channels=channels,
                        rate=sample_rate,
                        format=FORMAT,
                        periodsize=periodsize,
                        device=device_input)

    out = alsaaudio.PCM(device_output_capture,
                        channels=channels,
                        rate=sample_rate,
                        format=FORMAT,
                        periodsize=160,
                        device=device_output)

    # Execute indefinitely - cut with "CTRL+C"
    first_time = 1
    while True:
        loops = COUNTER_CACHE_LOOPS
        if first_time:
            time.sleep(1)
            first_time = 0
        while loops > 0:
            loops -= 1
            # Read data from device
            l, data = inp.read()
            if l:  # only if mic is open
                pass

                # chunk the data https://stackoverflow.com/a/33246354/10491422
                # packedData = map(lambda v: struct.pack('h', v), data)
                # frames = b''.join(packedData)
                # output_wave('temporal/input.wav', frames)

                audio_cache+=data
                # if loops:
                # print(loops)
                packedData = map(lambda v: struct.pack('h', v), audio_cache)
                frames = b''.join(packedData)
                output_wave('temporal/input.wav', frames)


                # Process in chunks
                # executeprediction('speechenhancement')


                # play as audio
                out.write(data)


