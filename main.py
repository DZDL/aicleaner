# 1. ALL LIBRARIES
# Import models: here we add new AI models.
# from aimodels.speechenhancement.speechenhancement import load_model, prediction # dev
from aimodels.speechenhancement.speechenhancement_production import prediction_production  # production

import os
import sys
import threading  # to manage threads
import time
import wave  # to read and write audio files

import numpy as np  # to treat audio data as numpy array
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
from rich.console import Console  # For colorful output

if (sys.platform == 'linux'):
    import alsaaudio  # input/output only for Linux ALSA

"""
2. GLOBAL CONFIGS.
"""
console = Console()  # Create object to print colorful
console.print("[GLOBAL CONFIGS] Loading variables and constants...",
              style="bold green")
# channels = 1  # Mono
sample_rate = 44100  # Sample read to read and write
# periodsize = 320
# FORMAT = alsaaudio.PCM_FORMAT_S16_LE
# COUNTER_CACHE_LOOPS = 1000
chunk = 1024  # set the chunk size of 1024 samples
FORMAT_INPUT = pyaudio.paInt32  # default LE_32bits or pyaudio.paInt32
FORMAT_OUTPUT = pyaudio.paFloat32  # default LE_32bits or pyaudio.paFloat32
channels = 1  # default mono
sample_rate_input = 44100  # default 44100
sample_rate_output = 8000  # default 8000 due restrictions, can't be more by 2021-03-02
record_seconds = 1.5  # default 1.1 due some restrictions
x = 0
y = 0
process_queue = []  # ready to process
player_queue = []  # ready to play
console.print("[GLOBAL CONFIGS] Finish loading variables and constants.",
              style="bold green")

"""
3. FUNCTIONS
"""


def hello():
    console.print("""|---------------------------------------------------------------------------------------------|
|  ______   ______   ______   __        ________   ______   __    __  ________  _______       |
| /      \ |      \ /      \ |  \      |        \ /      \ |  \  |  \|        \|       \      |
||  $$$$$$\ \$$$$$$|  $$$$$$\| $$      | $$$$$$$$|  $$$$$$\| $$\ | $$| $$$$$$$$| $$$$$$$\     |
|| $$__| $$  | $$  | $$   \$$| $$      | $$__    | $$__| $$| $$$\| $$| $$__    | $$__| $$     |
|| $$    $$  | $$  | $$      | $$      | $$  \   | $$    $$| $$$$\ $$| $$  \   | $$    $$     |
|| $$$$$$$$  | $$  | $$   __ | $$      | $$$$$   | $$$$$$$$| $$\$$ $$| $$$$$   | $$$$$$$\     |
|| $$  | $$ _| $$_ | $$__/  \| $$_____ | $$_____ | $$  | $$| $$ \$$$$| $$_____ | $$  | $$     |
|| $$  | $$|   $$ \ \$$    $$| $$     \| $$     \| $$  | $$| $$  \$$$| $$     \| $$  | $$     |
|\ $$   \$$ \$$$$$$  \$$$$$$  \$$$$$$$$ \$$$$$$$$ \$$   \$$ \$$   \$$ \$$$$$$$$ \$$   \$$     |
|                                                                                             |
|Mantainer: Pablo Diaz (github.com/zurmad)                                                    |
|Github: https://github.com/DZDL/aicleaner                                                    |
|License: MIT                                                                                 |
|Version: Alpha v0.1 [2020-03-08]                                                             |
|Operative System detected: {}
|Branch: Multithreading - 3 main threads (process thread can increase)                        |
|---------------------------------------------------------------------------------------------|""".format(sys.platform), style="bold yellow")


def clean_temporal_files():
    """
    Remove all files in specific paths
    """
    paths_to_remove = ['temporal/']

    console.print("[GLOBAL CONFIGS] Cleaning temporal files in {}".format(paths_to_remove),
                  style="bold green")

    try:
        for mypath in paths_to_remove:
            for f in os.listdir(mypath):
                if not 'README' in f:
                    os.remove(os.path.join(mypath, f))
    except Exception as e:
        print(e)

    console.print("[GLOBAL CONFIGS] Finish cleaning temporal files in {}".format(paths_to_remove),
                  style="bold green")


def load_input_device():
    """
    This function initialize mic input
    object and returns the objects get data
    """
    console.print("[GLOBAL CONFIGS] Loading input devices...",
                  style="bold green")
    p = pyaudio.PyAudio()  # initialize PyAudio object
    stream = p.open(format=FORMAT_INPUT,
                    channels=channels,
                    rate=sample_rate_input,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)  # open stream object as input & output
    console.print("[GLOBAL CONFIGS] Finish loading input devices.",
                  style="bold green")
    return p, stream


def load_output_device():
    """
    This function initialize the speakers output
    object and then return it to write data into it.
    """
    console.print("[GLOBAL CONFIGS] Loading output devices...",
                  style="bold green")

    if (sys.platform == 'linux'):
        device = alsaaudio.PCM(channels=1,
                               rate=sample_rate_output,
                               format=32,
                               periodsize=1000,
                               device='default')

    console.print("[GLOBAL CONFIGS] Finish loading output devices.",
                  style="bold green")

    return device


def record_one_second(filename_path='temporal/input.wav', thread_name=None, p=None, stream=None):
    """
    This script record around one second with pyaudio with specific finename_path
    Example taken from https://www.thepythoncode.com/article/play-and-record-audio-sound-in-python
    """
    console.print("{}[RECORD] Recording on {}".format(
        thread_name, filename_path), style="bold red")
    frames = []

    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)

    # stream.stop_stream()  # stop and close stream
    # stream.close()
    # p.terminate()  # terminate pyaudio object

    # save audio file in 'write bytes' mode
    wf = wave.open(filename_path, "wb")
    wf.setnchannels(channels)  # set the channels
    wf.setsampwidth(p.get_sample_size(FORMAT_INPUT))  # set the sample format
    wf.setframerate(sample_rate)  # set the sample rate
    wf.writeframes(b"".join(frames))  # write the frames as bytes
    wf.close()  # close the file
    console.print("{}[RECORD] Finished recording on {}".format(
        thread_name, filename_path), style="bold red")


def executeprediction(aimodel='speechenhancement', number=-1, thread_name=None, output=None):
    """
    This function is a manager of ai models applied
    like filters to noisy mic
    """

    # Speech-Enhancement model
    if aimodel == 'speechenhancement':
        console.print("{}[AI MODEL] {} -[INFERENCE]-cleaning on {}".format(thread_name,
                                                                           aimodel,
                                                                           'temporal/input_{}.wav'.format(number)),
                      style="bold red")
        prediction_production(weights_path='aimodels/speechenhancement/weights',
                              name_model='model_unet',
                              audio_dir_prediction='temporal',
                              dir_save_prediction='temporal/',
                              audio_input_prediction=[
                                  'input_{}.wav'.format(number)],
                              audio_output_prediction='output_{}.wav'.format(
                                  number),
                              sample_rate=8000,  # default 8000
                              min_duration=1.0,  # default 1.1
                              frame_length=8064,  # default 8064
                              hop_length_frame=8064,  # default 8064
                              n_fft=255,  # default 255
                              hop_length_fft=63,  # default 63
                              output=output)
        # os.remove('temporal/input_{}.wav')
        console.print("{}[AI MODEL] Speech-Enhancement finished. Saving {}".format(thread_name,
                                                                                   'temporal/output_{}.wav'.format(number)),
                      style="bold red")
    # Here you can add a new models
    else:
        console.print("{}[AI MODEL] No aimodel selected".format(thread_name,))
        return False


def play_one_file(filename_path='temporal/output.wav', thread_name=None, device=None):
    """
    This script play file
    Example adapted from https://people.csail.mit.edu/hubert/pyaudio/
    """
    console.print("{}[PLAY] Playing {}".format(thread_name,
                                               filename_path),
                  style="bold red")
    if (sys.platform == 'windows'):
        """
        Only Windows support
        """
        # sound = AudioSegment.from_file(filename_path)
        # play(sound)
        # Uncomplete
        pass
    elif (sys.platform == 'macos'):
        """
        Only macos support
        """
        pass
    elif (sys.platform == 'linux'):
        """
        Only linux support
        """
        with wave.open(filename_path, 'rb') as f:
            periodsize = 1000
            # Read data
            data = f.readframes(periodsize)
            while data:
                # Read data from stdin
                device.write(data)
                data = f.readframes(periodsize)

    console.print("{}[PLAY] Finish audio playing {}".format(thread_name,
                                                            filename_path),
                  style="bold red")


def record_only_thread(p, stream):

    global x
    global process_queue

    thread_name = threading.current_thread().name

    while True:
        record_one_second(filename_path='temporal/input_{}.wav'.format(x),
                          thread_name=thread_name,
                          p=p,
                          stream=stream)
        process_queue.append(x)  # append to the queue
        console.print('{} RECORD: {}'.format(thread_name, str(process_queue)),
                      style="bold green")
        x += 1  # Last line, don't move


def process_only_thread(lock_process):

    global process_queue
    global player_queue

    thread_name = threading.current_thread().name

    while True:
        if len(process_queue) > 0:
            # lock_process.acquire()
            # release the first queued but get the value

            number = process_queue.pop(0)
            console.print('{} PROCESS: {}'.format(thread_name,
                                                  str(number)),
                          style="bold green")

            # lock_process.release()
            executeprediction(aimodel='speechenhancement',
                              number=number,
                              thread_name=thread_name,
                              output=False)
            player_queue.append(number)


def play_only_thread(device):

    global player_queue

    thread_name = threading.current_thread().name
    while True:
        if len(player_queue) > 0:
            number = player_queue.pop(0)
            console.print('{} POPPED: {}'.format(thread_name,
                                                 str(number)),
                          style="bold green")
            play_one_file(filename_path='temporal/output_{}.wav'.format(number),
                          thread_name=thread_name,
                          device=device)


"""
4. LOOP ITERATION TO PROCESS DIRTY TO CLEAN AUDIO
"""

if __name__ == '__main__':

    # Initial configurations

    hello()  # Logo and info of the project
    clean_temporal_files()  # Clean files by path
    device = load_output_device()  # only for linux pyalsaaudio
    p, stream = load_input_device()  # only for linux pyalsaaudio
    x = 0
    y = 0
    process_queue = []  # ready to process
    player_queue = []  # ready to play
    lock_process = threading.Lock()

    time.sleep(2)  # drivers slow start needed
    # Record voice continously
    t1 = threading.Thread(target=record_only_thread,
                          args=(p, stream),
                          name='[T1]')

    # Process voice continously by queue
    t2 = threading.Thread(target=process_only_thread,
                          args=(lock_process,),
                          name='[T2]')

    # t4 = threading.Thread(target=process_only_thread,
    #                       args=(lock_process,),
    #                       name='[T4]')

    # Play voice continously by queue
    t3 = threading.Thread(target=play_only_thread,
                          args=(device,),
                          name='[T3]')

    time.sleep(2)  # drivers slow start needed
    # Charge and join threads
    t1.start()
    t2.start()
    t3.start()
    # t4.start()

    t1.join()
    t2.join()
    t3.join()
    # t4.join()
