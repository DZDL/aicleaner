
import os
import subprocess

import librosa
import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

weights_path='aimodels/speechenhancement/weights',
name_model='model_unet'
audio_dir_prediction='temporal'
dir_save_prediction='temporal/'
audio_input_prediction=['input.wav']
audio_output_prediction='output.wav'
sample_rate=8000  # default 8000
min_duration=1.1  # default 1.0 second
frame_length=8064  # default 8064
hop_length_frame=8064  # default 8064
n_fft=255  # default 255
hop_length_fft=63

# load json and create model
json_file = open('/home/god/Desktop/gits/aicleaner/aimodels/speechenhancement/weights/model_unet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('/home/god/Desktop/gits/aicleaner/aimodels/speechenhancement/weights/model_unet.h5')
print("Loaded model from disk")


import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    loaded_model,
    '/home/$USER/Desktop/gits/aicleaner/aimodels/speechenhancement/serving/1',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
)


# ./1/ exported path to make inferences