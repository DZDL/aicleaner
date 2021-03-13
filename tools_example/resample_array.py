import numpy as np
import samplerate

array_audio_example=np.asarray([[0, 1.1 ,2.2, 3.3, 4, 5, 6, 7 ,8 ,9.984]])

# downsample 1/2
sample_rate_output=1
sample_rate_input=2
y = samplerate.resample(array_audio_example[0], sample_rate_output * 1.0 / sample_rate_input, 'sinc_best')  
    
print(array_audio_example)

print("------------------------")

print(y)