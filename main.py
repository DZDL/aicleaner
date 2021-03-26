# 1. ALL LIBRARIES
# Import models: here we add new AI models.
# from aimodels.speechenhancement.speechenhancement import load_model, prediction # dev
import os
import sys
import time
import wave  # to read and write audio files
# import threading  # to manage threads
# to manage 3 tasks at the same time
from multiprocessing import Process, Queue, current_process

import numpy  # to treat audio data as numpy array
from rich.console import Console  # For colorful output

from aimodels.speechenhancement.speechenhancement_production import (  # production
    prediction_production,
    prediction_production_data_as_narray)

import soundfile as sf

"""
2. GLOBAL CONFIGS.
"""
console = Console()  # Create object to print colorful
console.print("[GLOBAL CONFIGS] Loading variables and constants...",
              style="bold green")
# channels = 1  # Mono
# sample_rate = 44100  # Sample read to read and write
# periodsize = 320
# FORMAT = alsaaudio.PCM_FORMAT_S16_LE
# COUNTER_CACHE_LOOPS = 1000
# chunk = 1024  # set the chunk size of 1024 samples
# FORMAT_INPUT = pyaudio.paInt32  # default LE_32bits or pyaudio.paInt32
# FORMAT_OUTPUT = pyaudio.paFloat32  # default LE_32bits or pyaudio.paFloat32
channels = 1  # default mono
sample_rate_input = 44100  # default 44100
sample_rate_output = 8000  # default 8000 due restrictions, can't be more by 2021-03-02
record_seconds = 1.2  # default 1.1 due some restrictions
INITIAL_DELAY_SECONDS = 0.5  # initial delay in seconds to create threads
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
|Version: Alpha v0.1 [2020-03-13]                                                             |
|Operative System detected: {}
|Branch: MultiProcessing - 4 main Process (Record, Process, Play, Server)                     |
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


def executeprediction(aimodel='speechenhancement',
                      number=-1,
                      process_name=None,
                      output=None,
                      asnumpyarray=False,
                      mydata=None):
    """
    This function is a manager of ai models applied
    like filters to noisy mic
    """

    # Speech-Enhancement model
    if aimodel == 'speechenhancement':
        console.print("{}[AI MODEL] {} -[INFERENCE]-cleaning on {}".format(process_name,
                                                                           aimodel,
                                                                           'temporal/input_{}.wav'.format(number)),
                      style="bold red")
        mydata_predicted = 0
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
                                                                    sample_rate=sample_rate_input,  # default 8000
                                                                    mydata=mydata,  # data as numpy
                                                                    output=output)

        console.print("{}[AI MODEL] Speech-Enhancement finished. Saving {}".format(process_name,
                                                                                   'temporal/output_{}.wav'.format(number)),
                      style="bold red")

        return mydata_predicted
    # Here you can add a new models
    else:
        console.print("{}[AI MODEL] No aimodel selected".format(process_name,))
        return False


def record_only_Process(queue_data_recorded,
                        queue_data_recorded_index):

    import soundcard as sc  # Exception due Exceptions

    process_name = current_process().name
    start_time = time.time()

    x = 0
    numframes = int(sample_rate_input*record_seconds)

    default_mic = sc.default_microphone()
    recorder = default_mic.recorder(sample_rate_input, channels=channels, blocksize=256)

    with recorder as r:
        while True:
            data = r.record(numframes)  # record each _x seconds_

            # Threading (Latency increased drastically)
            # data_queue.append(data)  # append data to the queue
            # process_queue.append(x)  # append index

            # MultiProcessing
            queue_data_recorded.put(data)
            queue_data_recorded_index.put(x)

            # sf.write('temporal/input_{}.wav'.format(x), data, sample_rate_input, 'PCM_32')

            console.print('{} RECORD: {}'.format(process_name, str(x)),
                          style="bold green")
            x += 1  # Last line, don't move


def process_only_Process(queue_data_recorded,
                         queue_data_recorded_index,
                         queue_data_to_play,
                         queue_data_to_play_index):

    time.sleep(INITIAL_DELAY_SECONDS)
    process_name = current_process().name

    while True:

        number = queue_data_recorded_index.get()
        if number != -1:
            # release the first queued but get the value
            mydata = queue_data_recorded.get()
            console.print('{} PROCESS: {}'.format(process_name,
                                                  str(number)),style="bold green")

            # With data as numpy array
            mydata_predicted = executeprediction(aimodel='speechenhancement',
                                                 number=number,
                                                 process_name=process_name,
                                                 output=None,
                                                 asnumpyarray=True,
                                                 mydata=mydata)
            queue_data_to_play_index.put(number)
            queue_data_to_play.put(mydata_predicted)
        else:
            time.sleep(0.05) # to don't stress cpu


def play_only_Process(queue_data_to_play,
                      queue_data_to_play_index):

    import soundcard as sc

    process_name = current_process().name

    time.sleep(INITIAL_DELAY_SECONDS*2)

    default_speaker = sc.default_speaker()
    default_speaker_player = default_speaker.player(sample_rate_output, channels=channels, blocksize=256)

    with default_speaker_player as p:
        while True:
            number = queue_data_to_play_index.get()
            if number != -1:
                mydata_predicted = queue_data_to_play.get()
                console.print('{} POPPED: {}'.format(process_name, str(number)),
                              style="bold green")

                # sf.write('temporal/output_{}.wav'.format(number),
                #          mydata_predicted, sample_rate_output, 'PCM_32')

                p.play(mydata_predicted/numpy.max(mydata_predicted))
            else:
                time.sleep(0.01) # to don't stress cpu


def power_on_tensorflow_serving_Process():

    api_port = 8501
    model_name = 'model_unet'
    model_base_path = os.path.dirname(os.path.abspath('main.py'))
    model_relative_path=model_base_path+'/aimodels/speechenhancement/serving/'

    command = 'tensorflow_model_server --rest_api_port={} --model_name={} --model_base_path="{}"'.format(api_port,
                                        model_name,
                                        model_relative_path)

    process_name = current_process().name

    console.print('{} TENSORFLOW SERVER TURNING ON...'.format(process_name),
                          style="bold green")
    # Turn on
    print(command)
    os.popen(command)

    console.print('{} TENSORFLOW SERVER READY.'.format(process_name),
                          style="bold green")

    # no return



"""
4. LOOP ITERATION TO PROCESS DIRTY TO CLEAN AUDIO
"""


if __name__ == '__main__':

    # Initial configurations

    hello()  # Logo and info of the project
    clean_temporal_files()  # Clean files by path

    queue_data_recorded_index = Queue()
    queue_data_recorded = Queue()

    queue_data_to_play_index = Queue()
    queue_data_to_play = Queue()

    time.sleep(INITIAL_DELAY_SECONDS)  # drivers slow start needed

    # Turn on server of tensorflow serving
    P0_SERVER_PROCESS=Process(target=power_on_tensorflow_serving_Process,name='[SERVER]')
    P0_SERVER_PROCESS.start()
    P0_SERVER_PROCESS.join()

    time.sleep(INITIAL_DELAY_SECONDS)  # drivers slow start needed

    # Record voice continously
    P1_RECORDER_PROCESS = Process(target=record_only_Process, args=((queue_data_recorded),
                                                 (queue_data_recorded_index)), name='[P1]')
    # Process as queued
    P2_PROCESSMODEL_PROCESS = Process(target=process_only_Process, args=((queue_data_recorded),
                                                    (queue_data_recorded_index), 
                                                    (queue_data_to_play),
                                                    (queue_data_to_play_index)), name='[P2]')
    # Play as finished processed
    P3_PLAYER_PROCESS = Process(target=play_only_Process, args=((queue_data_to_play), 
                                                 (queue_data_to_play_index)), name='[P3]')

    # Charge and join threads
    P1_RECORDER_PROCESS.start()
    P2_PROCESSMODEL_PROCESS.start()
    P3_PLAYER_PROCESS.start()

    P1_RECORDER_PROCESS.join()
    P2_PROCESSMODEL_PROCESS.join()
    P3_PLAYER_PROCESS.join()
