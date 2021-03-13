#!/usr/bin/env python3
# -*- mode: python; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-
# Taken from https://github.com/larsimmisch/pyalsaaudio/blob/master/playwav.py
# Simple test script that plays (some) wav files

from __future__ import print_function

import getopt
import sys
import wave
import time

import alsaaudio


# files = ['temporal/recorded_0.wav',
#          'temporal/recorded_1.wav',
#          'temporal/recorded_2.wav',
#          'temporal/recorded_3.wav',
#          'temporal/recorded_4.wav']

files_input = ['temporal/input_0.wav',
               'temporal/input_1.wav',
               'temporal/input_2.wav',
               'temporal/input_3.wav',
               'temporal/input_4.wav',
               'temporal/input_5.wav',
               'temporal/input_6.wav',
               'temporal/input_7.wav',]
            #    'temporal/input_8.wav',
            #    'temporal/input_9.wav',
            #    'temporal/input_10.wav',
            #    'temporal/input_11.wav',
            #    'temporal/input_12.wav',
            #    'temporal/input_13.wav',
            #    'temporal/input_14.wav',
            #    'temporal/input_15.wav',
            #    'temporal/input_16.wav',
            #    'temporal/input_17.wav',
            #    'temporal/input_18.wav',
            #    'temporal/input_19.wav',
            #    'temporal/input_20.wav', ]

files_output = ['temporal/output_0.wav',
                'temporal/output_1.wav',
                'temporal/output_2.wav',
                'temporal/output_3.wav',
                'temporal/output_4.wav',
                'temporal/output_5.wav',
                'temporal/output_6.wav',
                'temporal/output_7.wav',
                'temporal/output_8.wav',
                'temporal/output_9.wav',
                'temporal/output_10.wav',
                'temporal/output_11.wav',
                'temporal/output_12.wav',
                'temporal/output_13.wav',
                'temporal/output_14.wav',
                'temporal/output_15.wav',
                'temporal/output_16.wav',
                'temporal/output_17.wav',
                'temporal/output_18.wav',
                'temporal/output_19.wav',
                'temporal/output_20.wav', ]

device_output = alsaaudio.PCM(channels=1,
                       rate=8000,
                       format=32,
                       periodsize=1000,
                       device='default')

device_input = alsaaudio.PCM(channels=1,
                       rate=44100,
                       format=alsaaudio.PCM_FORMAT_S32_LE,
                       periodsize=44100//8,
                       device='default')

device=device_input

def play(f):

    global device

    # format = None

    # # 8bit is unsigned in wav files
    # if f.getsampwidth() == 1:
    #     format = alsaaudio.PCM_FORMAT_U8
    # # Otherwise we assume signed data, little endian
    # elif f.getsampwidth() == 2:
    #     format = alsaaudio.PCM_FORMAT_S16_LE
    # elif f.getsampwidth() == 3:
    #     format = alsaaudio.PCM_FORMAT_S24_3LE
    # elif f.getsampwidth() == 4:
    #     format = alsaaudio.PCM_FORMAT_S32_LE
    # else:
    #     raise ValueError('Unsupported format')

    periodsize = f.getframerate() // 8

    # print('%d channels, %d sampling rate, format %d, periodsize %d\n' % (1,
                                                                        #  44100,
                                                                        #  format,
                                                                        #  periodsize))

    data = f.readframes(periodsize)
    if data:
        while data:
            # Read data from stdin
            if data:
                device.write(data)
                # print(data)
                data = f.readframes(periodsize)
    else:
        mydata=bytes('\x00', 'utf-8')
        device.write(mydata)
        print(str(mydata)+'-----------------------------------')


if __name__ == '__main__':

    for myfile in files_input:
        print("Playing {}".format(myfile))
        with wave.open(myfile, 'rb') as f:
            play(f)

        time.sleep(2)
