import os
import sys
import threading
import time
import wave

import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.playback import play



from rich.console import Console
console = Console()

# Import models: here we add new AI models.
"""
1.1 Develop mode
"""
#from aimodels.speechenhancement.speechenhancement import load_model, prediction
"""
1.2 Production mode
"""
from aimodels.speechenhancement.speechenhancement_production import prediction_production


"""
2. GLOBAL CONFIGS.
"""
console.print("[GLOBAL CONFIGS] Loading...", style="bold green")
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
record_seconds = 2.5  # default 1.1 due some restrictions
x = 0
y=0
console.print("[GLOBAL CONFIGS] Completed.", style="bold green")

"""
3. FUNCTIONS
"""


def record_one_second(filename_path='temporal/input.wav',thread_name=''):
    """
    This script record around one second with pyaudio with specific finename_path
    Example taken from https://www.thepythoncode.com/article/play-and-record-audio-sound-in-python
    """
    console.print("{}[RECORD] Recording on {}".format(thread_name,filename_path), style="bold red")
    p = pyaudio.PyAudio()  # initialize PyAudio object
    stream = p.open(format=FORMAT_INPUT,
                    channels=channels,
                    rate=sample_rate_input,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)  # open stream object as input & output
    frames = []
    
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    

    stream.stop_stream()  # stop and close stream
    stream.close()
    p.terminate()  # terminate pyaudio object

    # save audio file in 'write bytes' mode
    wf = wave.open(filename_path, "wb")
    wf.setnchannels(channels)  # set the channels
    wf.setsampwidth(p.get_sample_size(FORMAT_INPUT))  # set the sample format
    wf.setframerate(sample_rate)  # set the sample rate
    wf.writeframes(b"".join(frames))  # write the frames as bytes
    wf.close()  # close the file
    console.print("{}[RECORD] Finished recording on {}".format(thread_name,filename_path), style="bold red")


def executeprediction(aimodel='speechenhancement', number=-1,thread_name=''):
    """
    This function is a manager of ai models applied
    like filters to noisy mic
    """

    # Speech-Enhancement model
    if aimodel == 'speechenhancement':
        console.print(
            "{}[AI MODEL] {} -[INFERENCE]-cleaning on {}".format(thread_name,aimodel,'temporal/input_{}.wav'.format(number)), style="bold red")
        prediction_production(weights_path='aimodels/speechenhancement/weights',
                              name_model='model_unet',
                              audio_dir_prediction='temporal',
                              dir_save_prediction='temporal/',
                              audio_input_prediction=[
                                  'input_{}.wav'.format(number)],
                              audio_output_prediction='output_{}.wav'.format(
                                  number),
                              sample_rate=8000,  # default 8000
                              min_duration=1.1,  # default 1.1
                              frame_length=8064,  # default 8064
                              hop_length_frame=8064,  # default 8064
                              n_fft=255,  # default 255
                              hop_length_fft=63)  # default 63
        # os.remove('temporal/input_{}.wav')
        console.print("{}[AI MODEL] Speech-Enhancement finished. Saving {}".format(thread_name,'temporal/output_{}.wav'.format(number)), style="bold red")
    # Here you can add a new models
    else:
        console.print("{}[AI MODEL] No aimodel selected".format(thread_name,))
        return False

def play_one_second(filename_path='temporal/output.wav',thread_name=''):
    """
    This script play file
    Example adapted from https://people.csail.mit.edu/hubert/pyaudio/
    """
    console.print("{}[PLAY] Playing {}".format(thread_name,filename_path), style="bold red")
    if (sys.platform == 'windows' or
            sys.platform == 'macos'):
        sound = AudioSegment.from_file(filename_path)
        play(sound)
    elif (sys.platform == 'linux'):
        command_inference = 'python3 tools_example/pyalsaaudio_playback.py {}'.format(filename_path)
        result = os.popen(command_inference).read()
    console.print("{}[PLAY] Finish audio playing {}".format(thread_name,filename_path), style="bold red")

def record_process_play_thread(lock_record, lock_playing):

    global x
    global y

    x=0
    y=0

    thread_name=threading.current_thread().name

    while True:

        lock_record.acquire()
        record_one_second(filename_path='temporal/input_{}.wav'.format(x),thread_name=thread_name)
        x+=1 
        lock_record.release( )

        lock_playing.acquire()     
        executeprediction(aimodel='speechenhancement', number=x-1,thread_name=thread_name)   
        play_one_second(filename_path='temporal/output_{}.wav'.format(y),thread_name=thread_name)
        y+=1
        lock_playing.release()


def hello():
    console.print("""------------------------------------------------------------------------------------
 ______   ______   ______   __        ________   ______   __    __  ________  _______  
/      \ |      \ /      \ |  \      |        \ /      \ |  \  |  \|        \|       \ 
|  $$$$$$\ \$$$$$$|  $$$$$$\| $$      | $$$$$$$$|  $$$$$$\| $$\ | $$| $$$$$$$$| $$$$$$$\ 
| $$__| $$  | $$  | $$   \$$| $$      | $$__    | $$__| $$| $$$\| $$| $$__    | $$__| $$
| $$    $$  | $$  | $$      | $$      | $$  \   | $$    $$| $$$$\ $$| $$  \   | $$    $$
| $$$$$$$$  | $$  | $$   __ | $$      | $$$$$   | $$$$$$$$| $$\$$ $$| $$$$$   | $$$$$$$\ 
| $$  | $$ _| $$_ | $$__/  \| $$_____ | $$_____ | $$  | $$| $$ \$$$$| $$_____ | $$  | $$
| $$  | $$|   $$ \ \$$    $$| $$     \| $$     \| $$  | $$| $$  \$$$| $$     \| $$  | $$
\ $$   \$$ \$$$$$$  \$$$$$$  \$$$$$$$$ \$$$$$$$$ \$$   \$$ \$$   \$$ \$$$$$$$$ \$$   \$$
                                                                                        
                                                                                        """, style="bold red")


"""
4. LOOP ITERATION TO PROCESS DIRTY TO CLEAN AUDIO
"""

if __name__ == '__main__':

    lock_record = threading.Lock()
    lock_playing = threading.Lock()

    hello()

    # start_time = time.time()
    # Record voice and generate temporal/input_x.wav, process and play
    t1 = threading.Thread(target=record_process_play_thread, args=(
        lock_record, lock_playing), name='[T1]')
    t2 = threading.Thread(target=record_process_play_thread, args=(
        lock_record, lock_playing), name='[T2]')
    t3 = threading.Thread(target=record_process_play_thread, args=(
        lock_record, lock_playing), name='[T3]')
    t4 = threading.Thread(target=record_process_play_thread, args=(
        lock_record, lock_playing), name='[T4]')

    t1.start()
    t2.start()
    t3.start()
    t4.start()


    t1.join()
    t2.join()
    t3.join()
    t4.join()
