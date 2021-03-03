
from __future__ import print_function
import os
import sys
import time
import getopt
import alsaaudio

command_inference = 'cd audiodenoising && python3 main.py --mode "prediction" --audio_dir_prediction "input/"  --dir_save_prediction "output/" --audio_output_prediction "output.wav" --sample_rate 44100 --min_duration 0 && cd ..'


def usage():
    print('usage: recordtest.py [-d <device>] <file>', file=sys.stderr)
    sys.exit(2)


if __name__ == '__main__':

    device_input = 'sysdefault'
    device_input_capture = alsaaudio.PCM_CAPTURE
    device_input_type = alsaaudio.PCM_NONBLOCK
    device_output = 'default'
    device_output_capture = alsaaudio.PCM_PLAYBACK
    channels = 1
    sample_rate = 44100
    periodsize = 320
    FORMAT = alsaaudio.PCM_FORMAT_S32_LE
    COUNTER_LOOP=10000

    # Open the device in nonblocking capture mode in mono, with a sampling rate of 44100 Hz
    # and 16 bit little endian samples
    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    # For our purposes, it is suficcient to know that reads from the device
    # will return this many frames. Each frame being 2 bytes long.
    # This means that the reads below will return either 320 bytes of data
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # mode.
    inp = alsaaudio.PCM(device_input_capture, device_input_type, channels=channels,
                        rate=sample_rate, format=FORMAT, periodsize=periodsize, device=device_input)
    print(inp)

    out = alsaaudio.PCM(device_output_capture, channels=channels, rate=sample_rate,
                        format=FORMAT, periodsize=160, device=device_output)
    first_time = 1

    
    # f1 = open('trash.wav', 'wb')
    # f2 = open('trash.wav', 'rb')


    while True:
        loops = COUNTER_LOOP
        if first_time:
            time.sleep(1)
            first_time = 0
        while loops > 0:
            loops -= 1
            # Read data from device
            l, data = inp.read()
            if l: # only if mic is open
                pass
            
            # Process in chunks
        
            # result = os.popen(command_inference).read()


            # play as audio
            out.write(data)
