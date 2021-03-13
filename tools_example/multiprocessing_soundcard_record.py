# Taken from https://github.com/bastibe/SoundCard
import soundcard as sc
import soundfile as sf
import numpy
import time

# get the current default microphone on your system:
default_mic = sc.default_microphone()

# alternatively, get a `Recorder` and `Player` object
# and play or record continuously:
record_object=default_mic.recorder(44100, channels=1, blocksize=256)

print(record_object)

seconds=1.2
numframes=int(44100*seconds)
max_iterations=20
expected_time_array=numpy.arange(seconds,(max_iterations+1)*seconds,seconds)

my_audio_streaming=[]

with record_object as r:
    start_time=time.time()
    for number in range(max_iterations):
        
        data=r.record(numframes)
        my_audio_streaming.append(data)
        # sf.write('temporal/pysoundcard_record_continously_{}.wav'.format(number), data, 44100, 'PCM_32') # add more seconds, not recommended
        now=time.time()        
        print("[{}] \tReal: {} \tExpected: {} \tNumframes:{}".format(number, round(now-start_time,4),expected_time_array[number],numframes))