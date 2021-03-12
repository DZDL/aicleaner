# 1. ALL LIBRARIES
# Import models: here we add new AI models.
# from aimodels.speechenhancement.speechenhancement import load_model, prediction # dev
from aimodels.speechenhancement.speechenhancement_production import prediction_production, prediction_production_data_as_narray  # production

import os
import sys
import threading  # to manage threads
import time
import wave  # to read and write audio files

import numpy  # to treat audio data as numpy array
from rich.console import Console  # For colorful output
import soundcard as sc
import soundfile as sf

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
# FORMAT_INPUT = pyaudio.paInt32  # default LE_32bits or pyaudio.paInt32
# FORMAT_OUTPUT = pyaudio.paFloat32  # default LE_32bits or pyaudio.paFloat32
channels = 1  # default mono
sample_rate_input = 44100  # default 44100
sample_rate_output = 8000  # default 8000 due restrictions, can't be more by 2021-03-02
record_seconds = 2  # default 1.1 due some restrictions
x = 0  # index of record iteration
y = 0  # index of play iteration
process_queue = []  # ready to process
data_queue = []  # numpy array inside an array ready to process
player_queue = []  # ready to play
data_predicted_queue = []  # numpy array inside an array ready to be played
INITIAL_DELAY_SECONDS = 1  # initial delay in seconds to create threads
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

    # get the current default microphone on your system:
    default_mic = sc.default_microphone()
    recorder = default_mic.recorder(44100, channels=1, blocksize=256)

    console.print("[GLOBAL CONFIGS] Finish loading input devices.",
                  style="bold green")
    return recorder


def load_output_device():
    """
    This function initialize the speakers output
    object and then return it to write data into it.
    """
    console.print("[GLOBAL CONFIGS] Loading output devices...",
                  style="bold green")

    # get the current default speaker on your system:
    default_speaker = sc.default_speaker()
    player=default_speaker.player(48000, channels=2, blocksize=512)
    

    console.print("[GLOBAL CONFIGS] Finish loading output devices.",
                  style="bold green")

    return player


def record_one_second(filename_path='temporal/input.wav',
                      thread_name=None,
                      default_mic=None,
                      asnumpyarray=False):
    """
    This script record around one second with soundcard
    """
    console.print("{}[RECORD] Recording on {}".format(
        thread_name, filename_path), style="bold red")

    data = default_mic.record(numframes=int(sample_rate_input*record_seconds),
                              samplerate=sample_rate_input, channels=channels)

    console.print("{}[RECORD] Finished recording on {}".format(
        thread_name, filename_path), style="bold red")

    if asnumpyarray == False:
        # save audio file in 'write bytes' mode
        sf.write(filename_path, data, sample_rate_input, 'PCM_32')
    elif asnumpyarray == True:
        # sf.write(filename_path, data, sample_rate_input, 'PCM_32')
        return data


def executeprediction(aimodel='speechenhancement',
                      number=-1,
                      thread_name=None,
                      output=None,
                      asnumpyarray=False,
                      mydata=None):
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
        mydata_predicted=0
        if (asnumpyarray == False):
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

        elif (asnumpyarray == True):
            mydata_predicted = prediction_production_data_as_narray(frame_length=8064,  # default 8064
                                                                    hop_length_frame=8064,  # default 8064
                                                                    n_fft=255,  # default 255
                                                                    hop_length_fft=63,  # default 63
                                                                    mydata=mydata,  # data as numpy
                                                                    output=output)

        console.print("{}[AI MODEL] Speech-Enhancement finished. Saving {}".format(thread_name,
                                                                                   'temporal/output_{}.wav'.format(number)),
                      style="bold red")

        return mydata_predicted
    # Here you can add a new models
    else:
        console.print("{}[AI MODEL] No aimodel selected".format(thread_name,))
        return False


def play_one_file(filename_path='temporal/output.wav',
                  thread_name=None,
                  device=None,
                  asnumpyarray=False,
                  audiodata=None):
    """
    This script play file
    Example adapted from https://people.csail.mit.edu/hubert/pyaudio/
    """
    console.print("{}[PLAY] Playing {}".format(thread_name,
                                               filename_path),
                  style="bold red")
    if (asnumpyarray == False):
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
    elif (asnumpyarray == True):
        default_speaker.play(audiodata/numpy.max(audiodata),
                             samplerate=8000)

    console.print("{}[PLAY] Finish audio playing {}".format(thread_name,
                                                            filename_path),
                  style="bold red")


def record_only_thread(default_mic_recorder):

    global x
    global process_queue
    global data_queue
    global record_seconds

    thread_name = threading.current_thread().name
    start_time = time.time()

    numframes = int(44100*record_seconds)

    with default_mic_recorder as r:
        while True:
            data = r.record(numframes)  # record each _x seconds_
            data_queue.append(data)  # append data to the queue
            process_queue.append(x)  # append index
            console.print('{} RECORD: {}'.format(thread_name, str(process_queue)),
                          style="bold green")
            x += 1  # Last line, don't move


def process_only_thread():

    global process_queue
    global player_queue
    global data_queue
    global INITIAL_DELAY_SECONDS

    time.sleep(INITIAL_DELAY_SECONDS)

    thread_name = threading.current_thread().name

    while True:
        if len(data_queue) > 0:
            # release the first queued but get the value
            number = process_queue.pop(0)
            mydata = data_queue.pop(0)
            console.print('{} PROCESS: {}'.format(thread_name,
                                                  str(number)),
                          style="bold green")

            # With wav file
            # executeprediction(aimodel='speechenhancement',
            #                   number=number,
            #                   thread_name=thread_name,
            #                   output=False)  # change to True to see other outputs

            # With data as numpy array
            mydata_predicted = executeprediction(aimodel='speechenhancement',
                                                 number=number,
                                                 thread_name=thread_name,
                                                 output=None,
                                                 asnumpyarray=True,
                                                 mydata=mydata)
            player_queue.append(number)
            data_predicted_queue.append(mydata_predicted)
        else:
            time.sleep(0.05)


def play_only_thread(default_speaker_player):

    global player_queue

    thread_name = threading.current_thread().name

    time.sleep(INITIAL_DELAY_SECONDS*2)

    with default_speaker_player as p:
        while True:
            if len(player_queue) > 0:
                number = player_queue.pop(0)
                mydata_predicted = data_predicted_queue.pop(0)
                console.print('{} POPPED: {}'.format(thread_name,
                                                    str(number)),
                            style="bold green")

                p.play(mydata_predicted/numpy.max(mydata_predicted))
                # play_one_file(filename_path='temporal/output_{}.wav'.format(number),
                #             thread_name=thread_name,
                #             device=p,
                #             asnumpyarray=True,
                #             audiodata=mydata_predicted)
            else:
                time.sleep(0.05)


"""
4. LOOP ITERATION TO PROCESS DIRTY TO CLEAN AUDIO
"""

if __name__ == '__main__':

    # Initial configurations

    hello()  # Logo and info of the project
    clean_temporal_files()  # Clean files by path
    default_speaker_player = load_output_device()
    default_mic_recorder = load_input_device()
    x = 0
    y = 0
    process_queue = []  # ready to process
    player_queue = []  # ready to play
    # lock_process = threading.Lock()

    time.sleep(INITIAL_DELAY_SECONDS)  # drivers slow start needed
    # Record voice continously
    t1 = threading.Thread(target=record_only_thread,
                          args=(default_mic_recorder,),
                          name='[T1]')

    # Process voice continously by queue
    t2 = threading.Thread(target=process_only_thread,
                          name='[T2]')

    # t4 = threading.Thread(target=process_only_thread,
    #                       args=(lock_process,),
    #                       name='[T4]')

    # # Play voice continously by queue
    t3 = threading.Thread(target=play_only_thread,
                          args=(default_speaker_player,),
                          name='[T3]')

    # Charge and join threads
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    # t4.start()

    t3.join()
    # t4.join()
