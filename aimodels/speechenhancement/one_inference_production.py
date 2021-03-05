"""
Documentation v0.1
Algorithm name: Speech-enhancement
Paper: https://towardsdatascience.com/speech-enhancement-with-deep-learning-36a1991d3d8d
Repository: https://github.com/vbelz/Speech-enhancement
License: MIT
"""

import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from speechenhancement_production import load_model, prediction

"""
GLOBAL CONFIGS
"""
weights_path = 'aimodels/speechenhancement/weights'
name_model = 'model_unet'
audio_dir_prediction = 'temporal'
dir_save_prediction = 'temporal/'
audio_input_prediction = ['input.wav']
audio_output_prediction = 'output.wav'
sample_rate = 8000  # default 8000
min_duration = 1.0  # default 1.0 second
frame_length = 8064  # default 8064
hop_length_frame = 8064  # default 8064
n_fft = 255  # default 255
hop_length_fft = 63  # default 63

"""
INFERENCE
"""
if __name__ == '__main__':

    loaded_model = load_model(weights_path,
                              name_model,
                              audio_dir_prediction,
                              dir_save_prediction,
                              audio_input_prediction,
                              audio_output_prediction,
                              sample_rate, min_duration,
                              frame_length,
                              hop_length_frame,
                              n_fft,
                              hop_length_fft)

    prediction(weights_path,
               name_model,
               audio_dir_prediction,
               dir_save_prediction,
               audio_input_prediction,
               audio_output_prediction,
               sample_rate, min_duration,
               frame_length,
               hop_length_frame,
               n_fft,
               hop_length_fft,
               loaded_model)
