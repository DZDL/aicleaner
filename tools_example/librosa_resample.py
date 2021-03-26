import librosa
import soundfile as sf

# Get example audio file
filename = librosa.ex('trumpet')

data, samplerate = sf.read(filename, dtype='float32')
print(data)
print(data.shape)
data = data.T
print(data)
print(data.shape)
data_22k = librosa.resample(data, samplerate, 8000)
print(data_22k)
print(data_22k.shape)