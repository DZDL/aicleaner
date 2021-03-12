"""
Documentation v0.1
Algorithm name: Speech-enhancement
Paper: https://towardsdatascience.com/speech-enhancement-with-deep-learning-36a1991d3d8d
Repository: https://github.com/vbelz/Speech-enhancement
License: MIT
"""
# Don't forget to start tensorflow server

"""
tensorflow_model_server --rest_api_port=8501 \ 
--model_name=model_unet \ 
--model_base_path="/home/god/Desktop/gits/aicleaner/aimodels/speechenhancement/serving/"
"""

import json
import librosa
import numpy as np
import requests
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import model_from_json


try:
    # Calling from file path
    from data_tools import (audio_files_to_numpy,
                            audio_files_to_numpy_from_numpy, 
                            inv_scaled_ou,
                            matrix_spectrogram_to_numpy_audio,
                            numpy_audio_to_matrix_spectrogram, 
                            scaled_in)

except Exception as e:
    # Calling from other paths
    print(e)
    from .data_tools import (audio_files_to_numpy,
                             audio_files_to_numpy_from_numpy, 
                             inv_scaled_ou,
                             matrix_spectrogram_to_numpy_audio,
                             numpy_audio_to_matrix_spectrogram, 
                             scaled_in)


def predict_with_tensorflow_server(unwrapped_data, output):
    """
    This POST function works with a tensorflow server running
    Do an inference in the production way.
    """

    url = "http://localhost:8501/v1/models/model_unet:predict"
    data = json.dumps({"signature_name": "serving_default",
                       "instances": unwrapped_data.tolist()})
    if output == True:
        print("[TENSORFLOW SERVER] DATA SENDING....-------------------")
        print(data[:400]+'"continue...')

    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    if output == True:
        print("[TENSORFLOW SERVER] JSON RESPONSE ---------------------")
        print(json_response.text[:400]+'"continue...')
    predictions = json.loads(json_response.text)['predictions']
    myreturn = np.asarray(predictions)
    return myreturn

    #show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))


def prediction_production(weights_path,
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
                          output):

    # Extracting noise and voice from folder and convert to numpy
    # print(audio_dir_prediction,
    #       audio_input_prediction,
    #       sample_rate,
    #       frame_length,
    #       hop_length_frame,
    #       min_duration)
    audio = audio_files_to_numpy(audio_dir_prediction,
                                 audio_input_prediction,
                                 sample_rate,
                                 frame_length,
                                 hop_length_frame,
                                 min_duration)

    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    # print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(audio,
                                                                     dim_square_spec,
                                                                     n_fft,
                                                                     hop_length_fft)

    # global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    # Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    # Prediction using loaded network
    print("------------------LOADED MODEL-----------------")
    # print('X_in|'+str(X_in))
    X_pred = predict_with_tensorflow_server(X_in, output)
    # print('X_pred|'+str(X_pred))
    print("------------------END LOADED MODEL-----------------")
    # Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    # Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]
    # Reconstruct audio from denoised spectrogram and phase
    # print(X_denoise.shape)
    # print(m_pha_audio.shape)
    # print(frame_length)
    # print(hop_length_fft)
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise,
                                                             m_pha_audio,
                                                             hop_length_frame,
                                                             hop_length_fft)
    # Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    # Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(
        1, nb_samples * frame_length)*10
    # librosa.output.write_wav(dir_save_prediction + audio_output_prediction,
    #                          denoise_long[0, :],
    #                          sample_rate)
    sf.write(dir_save_prediction + audio_output_prediction,
             denoise_long[0, :], sample_rate, 'PCM_32')


def prediction_production_data_as_narray(frame_length,
                                         sample_rate,
                                         hop_length_frame,
                                         n_fft,
                                         hop_length_fft,
                                         mydata,
                                         output):

    audio = audio_files_to_numpy_from_numpy(audio_data_numpy=np.asarray([mydata]),
                                            sample_rate=sample_rate,
                                            frame_length=frame_length,
                                            hop_length_frame=hop_length_frame)
    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    # print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(audio,
                                                                     dim_square_spec,
                                                                     n_fft,
                                                                     hop_length_fft)

    # global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    # Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    # Prediction using loaded network
    print("------------------LOADED MODEL-----------------")
    X_pred = predict_with_tensorflow_server(X_in, output)
    print("------------------END PREDICTION MODEL-----------------")
    # Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    # Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]
    # Reconstruct audio from denoised spectrogram and phase
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise,
                                                             m_pha_audio,
                                                             frame_length,
                                                             hop_length_fft)
    # Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    # Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(
        1, nb_samples * frame_length)*10

    sf.write('temporal/output.wav', denoise_long[0, :], sample_rate, 'PCM_32')

    return denoise_long[0, :]
